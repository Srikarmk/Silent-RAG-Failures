"""Diagnostic only: how much of the alarm rate on H_query_topic_shift comes from repeated queries (#73).

    python experiments/permutation/repeats.py

This is not a detection method and nothing here changes how a detector scores. The
permutation test assumes the reference and window values are exchangeable draws; a window
that asks some queries several times breaks that, because the copies are not independent.
To size the effect, each topic-shift window of the run is rescored three ways:

  full    exactly as the runner scored it (checked against the stored scores)
  dedup   one row per distinct query asked (analysis only)
  pseudo  the same mechanism and repeat rate, but the "topic" is a random subset of the pool
          of the same size, so queries repeat and there is no topical shift

`pseudo` alarming above alpha means repeats alone move the rule. The index never changes in
this control, so a query asked again gets exactly its clean result.

Needs the cached embeddings from the main run. Writes repeats.csv and repeats.txt next to it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from silent_rag import data
from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY
from silent_rag.inject.controls import QueryTopicShift

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ALPHA = 0.05
LABEL = "H_query_topic_shift"
SPLITS = {"A": [0, 1, 2], "B": [3, 4, 5]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "dev_check.yaml"))
    parser.add_argument("--results", default=str(ROOT / "results" / "permutation" / "dev_check.parquet"))
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=runner.UniqueKeyLoader)
    runner.check_datasets(cfg["datasets"], final=False)
    stored = pd.read_parquet(args.results)
    runner.load_plugins()
    names = [d if isinstance(d, str) else next(iter(d)) for d in cfg["detectors"]]
    bases = [n for n in names if not n.endswith("_perm") and n + "_perm" in names]
    W, K, fc = cfg["window_size"], cfg.get("k", 10), cfg["failures"][LABEL]

    ds = data.load("nfcorpus")
    retriever = runner.default_retriever(cfg)
    state = runner.read_only(retriever.embed_corpus(ds))
    docs = runner.doc_tuples(state)
    clean_ids, clean_scores, clean_q = retriever.search(state, k=K, query_embeddings=state.query_embeddings)
    position = {q: i for i, q in enumerate(state.query_ids)}

    def observe(src):
        src = np.asarray(src)
        return runner.observe(state, src, [clean_ids[j] for j in src], clean_scores[src], clean_q[src], docs)

    rows, start = [], time.perf_counter()
    for seed in cfg["seeds"]:
        order = np.random.default_rng(seed).permutation(len(state.query_ids))
        ref_idx, pool = order[:W], order[W:]
        reference = observe(ref_idx)
        dets = {}
        for b in bases:
            for name in (b, b + "_perm"):
                dets[name] = REGISTRY[name]()
                dets[name].fit(reference)
        topic = QueryTopicShift(fc["n_topics"], fc["topic"])
        topic.query_pool = [state.query_ids[i] for i in pool]
        topic.attach(retriever, state)
        # same apply(), a random non-topical "topic" of the same size
        pseudo = QueryTopicShift(fc["n_topics"], fc["topic"])
        pseudo.query_pool, pseudo.pool_positions = topic.query_pool, topic.pool_positions
        pseudo.attached_doc_ids = topic.attached_doc_ids
        pseudo.topic_positions = np.random.default_rng([seed, 99]).choice(pool, len(topic.topic_positions), replace=False)
        for severity in fc["severities"]:
            for trial in range(cfg["trials"]):
                idx = np.random.default_rng([seed, trial]).choice(pool, W, replace=False)
                inj_seed = runner.injector_seed(seed, trial)
                for variant, injector in (("full", topic), ("pseudo", pseudo)):
                    damaged = injector.apply(state, severity, seed=inj_seed)
                    src = [position[damaged.query_ids[i]] for i in idx]
                    variants = {variant: src, **({"dedup": list(dict.fromkeys(src))} if variant == "full" else {})}
                    for v, s in variants.items():
                        o = observe(s)
                        row = {"seed": seed, "severity": severity, "trial": trial, "variant": v,
                               "n": len(s), "n_distinct": len(set(s))}
                        row.update({name: d.score(o) for name, d in dets.items()})
                        rows.append(row)
    out = pd.DataFrame(rows)
    lines = [f"rescored {len(out)} windows in {time.perf_counter() - start:.0f}s"]

    st = stored[stored.failure == LABEL].pivot_table(index=["seed", "severity", "trial"], columns="detector",
                                                      values="score")
    full = out[out.variant == "full"].set_index(["seed", "severity", "trial"])[st.columns]
    diff = float(np.abs(full.sort_index().to_numpy() - st.sort_index().to_numpy()).max())
    lines.append(f"max |full - stored| score: {diff}  (must be 0: 'full' is the runner's window)")
    lines.append("\ndistinct queries per window of 100:\n" + out[out.variant != "dedup"].groupby(
        ["variant", "severity"]).n_distinct.mean().round(1).to_string())
    for split, calibration in SPLITS.items():
        ev = out[~out.seed.isin(calibration)]
        res = []
        for b in bases:
            clean = stored[(stored.detector == b) & (stored.failure == "clean") & stored.seed.isin(calibration)].score
            threshold = float(np.quantile(clean, 1 - ALPHA))
            for (v, severity), g in ev.groupby(["variant", "severity"]):
                res.append({"detector": b, "variant": v, "severity": severity,
                            "percentile_rule": (g[b] > threshold).mean(),
                            "permutation_rule": (g[b + "_perm"] > 1 - ALPHA).mean()})
        table = pd.DataFrame(res).pivot_table(index=["detector", "severity"], columns="variant",
                                              values=["percentile_rule", "permutation_rule"])
        lines.append(f"\n== split {split} (calibrate on {calibration}): alarm rates on evaluation seeds ==\n"
                     + table.round(2).to_string())
    text = "\n".join(lines)
    print(text)
    out_dir = Path(args.results).parent
    out.to_csv(out_dir / "repeats.csv", index=False)
    (out_dir / "repeats.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
