"""Clean nDCG@10 CIs and the query window size (#4).

From the repository root:
    python experiments/baseline/window_size.py              # NFCorpus only
    python experiments/baseline/window_size.py --held-out   # + clean CIs on SciFact and FiQA
    python experiments/baseline/window_size.py --max-seq-length 512   # the original #39 numbers

The window-size analysis only ever runs on NFCorpus, since picking the window is
tuning. --held-out only adds the clean-baseline CI for SciFact and FiQA, nothing
gets chosen from those numbers.

Docs are truncated at --max-seq-length tokens, 256 by default (#5). Writes under
results/baseline/window_size/msl<length>/.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from silent_rag import data
from silent_rag.eval.ground_truth import bootstrap_ci, per_query_ndcg
from silent_rag.eval.window import detection_rate, min_detectable_drop
from silent_rag.pipeline import Retriever
from silent_rag.run import check_datasets

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "baseline" / "window_size"
MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MODEL_REVISION = "b207367332321f8e44f96e224ef15bc607f4dbf0"
# absolute nDCG@10 drops. NFCorpus sits around 0.3, so 0.15 is already half of it.
DROPS = [round(d, 3) for d in np.arange(0, 0.1525, 0.005)]


def clean_scores(name: str, retriever: Retriever) -> tuple[np.ndarray, float]:
    ds = data.load(name)
    start = time.time()
    state = retriever.embed_corpus(ds)
    ids, _, _ = retriever.search(state, state.query_texts, k=10)
    return per_query_ndcg(ids, state.query_ids, ds.qrels, k=10), time.time() - start


def window_table(scores: np.ndarray, windows, churns, trials: int, seed: int) -> pd.DataFrame:
    """Detection rate for every (window, churn, drop). Windows are drawn the way
    run.py draws them: one reference window comes out first, the rest is the pool."""
    order = np.random.default_rng(seed).permutation(len(scores))
    rows = []
    for w in windows:
        pool = scores[order[w:]]
        if len(pool) < w:
            print(f"  window {w}: skipped, {len(scores)} queries can't fit a reference window and a pool of {w}")
            continue
        for churn in churns:
            for drop in DROPS:
                rate = detection_rate(pool, w, drop, trials=trials, churn=churn, seed=seed)
                rows.append({"window": w, "pool": len(pool), "churn": churn, "drop": drop, "rate": rate})
            print(f"  window {w}, churn {churn}: done")
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, baseline: float) -> list[dict]:
    out = []
    for (w, churn), g in table.groupby(["window", "churn"]):
        rates = dict(zip(g["drop"], g["rate"]))
        mdd80 = min_detectable_drop(rates, 0.8)
        out.append({
            "window": int(w),
            "pool": int(g["pool"].iloc[0]),
            "churn": float(churn),
            "false_alarm_rate": rates[0.0],
            "min_drop_50pct": min_detectable_drop(rates, 0.5),
            "min_drop_80pct": mdd80,
            "min_drop_80pct_relative": mdd80 / baseline,
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-out", action="store_true", help="also compute clean CIs on SciFact and FiQA")
    parser.add_argument("--windows", type=int, nargs="+", default=[25, 50, 100, 150, 200])
    parser.add_argument("--churn", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=256, help="doc truncation in tokens, 256 per #5")
    args = parser.parse_args()

    retriever = Retriever(MODEL, revision=MODEL_REVISION, max_seq_length=args.max_seq_length)
    datasets = ["nfcorpus"] + (["scifact", "fiqa"] if args.held_out else [])
    check_datasets(datasets, final=args.held_out)
    out = OUT / f"msl{args.max_seq_length}"
    out.mkdir(parents=True, exist_ok=True)

    baselines = {}
    per_query = {}
    for name in datasets:
        scores, seconds = clean_scores(name, retriever)
        mean, lo, hi = bootstrap_ci(scores)
        per_query[name] = scores
        baselines[name] = {
            "n_queries": len(scores),
            "ndcg@10": mean,
            "ci95": [lo, hi],
            "ci_half_width": (hi - lo) / 2,
            "zero_ndcg_fraction": float((scores == 0).mean()),
            "seconds": round(seconds, 1),
        }
        print(f"{name}: nDCG@10 = {mean:.4f} [{lo:.4f}, {hi:.4f}] over {len(scores)} queries")
        pd.DataFrame({"ndcg@10": scores}).to_csv(out / f"{name}_per_query.csv", index_label="position")

    print("window sizes on nfcorpus")
    table = window_table(per_query["nfcorpus"], args.windows, args.churn, args.trials, args.seed)
    table.to_csv(out / "nfcorpus_detection.csv", index=False)
    summary = summarize(table, baselines["nfcorpus"]["ndcg@10"])

    record = {
        "exploratory": True,
        "computed_on": date.today().isoformat(),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "retriever": retriever.config,
        "baselines": baselines,
        "window_analysis": {
            "dataset": "nfcorpus",
            "damage": "knockout to 0 with p = drop / window mean, after churn",
            "test": "paired_bootstrap, 2000 resamples, one-sided, alpha 0.05",
            "trials": args.trials,
            "seed": args.seed,
            "drops": DROPS,
            "summary": summary,
        },
        "ci_method": "percentile bootstrap over queries, 10000 resamples, seed 0",
        "platform": platform.platform(),
    }
    (out / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
