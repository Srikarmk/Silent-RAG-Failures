"""Experiment runner.

    python -m silent_rag.run --config configs/dev.yaml
    python -m silent_rag.run --config configs/final.yaml --final
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pkgutil
import time
from pathlib import Path

import numpy as np
import yaml

from silent_rag import data, detect, inject
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.eval.ground_truth import paired_bootstrap, per_query_ndcg
from silent_rag.eval.results import ResultRow, to_frame
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.pipeline import Retriever
from silent_rag.types import Observation

# filled in before anything reads the config, so run_id doesn't depend on whether a default was spelled out
DEFAULTS = {"k": 10, "seed": 0, "alpha": 0.05, "min_drop": 0.0}

DEV_DATASETS = {"nfcorpus"}
HELD_OUT = {"scifact", "fiqa"}
ALIASES = {"fiqa-2018": "fiqa"}


def canonical(name: str) -> str:
    name = name.strip().lower().removeprefix("beir/")
    return ALIASES.get(name, name)


def check_datasets(datasets: list[str], final: bool) -> list[str]:
    """We only tune on NFCorpus. Touching the held-out sets needs --final.

    Names are normalized first so "SciFact" or "BeIR/fiqa-2018" can't slip past.
    """
    names = list(dict.fromkeys(canonical(d) for d in datasets))
    unknown = set(names) - DEV_DATASETS - HELD_OUT
    if unknown:
        raise SystemExit(f"unknown datasets {sorted(unknown)}, expected one of {sorted(DEV_DATASETS | HELD_OUT)}")
    held_out = HELD_OUT.intersection(names)
    if held_out and not final:
        raise SystemExit(
            f"{sorted(held_out)} are held out. Tune on NFCorpus, and pass --final "
            "only for the real evaluation run."
        )
    return names


def load_plugins() -> None:
    for pkg in (inject, detect):
        for mod in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg.__name__}.{mod.name}")


def lookup(registry: dict, names, kind: str) -> dict:
    missing = [n for n in names if n not in registry]
    if missing:
        raise SystemExit(f"no {kind} registered for {missing}, available: {sorted(registry)}")
    return {n: registry[n] for n in names}


def run_id(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def check_severities(failures: dict) -> None:
    for f, sevs in failures.items():
        if not isinstance(sevs, list):
            raise SystemExit(f"{f}: severities must be a list, got {sevs!r}")

    def ok(s):
        return isinstance(s, (int, float)) and not isinstance(s, bool) and 0 < s <= 1

    bad = {f: s for f, sevs in failures.items() for s in sevs if not ok(s)}
    if bad:
        raise SystemExit(f"severity is the fraction affected and has to be in (0, 1], got {bad}")


def window_queries(retriever, clean, damaged, idx) -> np.ndarray:
    if damaged.query_embeddings is None:
        return retriever.embed_queries([damaged.query_texts[i] for i in idx])
    changed = [i for i in idx if damaged.query_texts[i] != clean.query_texts[i]]
    if changed and np.array_equal(damaged.query_embeddings[changed], clean.query_embeddings[changed]):
        raise RuntimeError(
            "injector changed query_texts but kept the clean query_embeddings, set them to None"
        )
    return damaged.query_embeddings[idx]


def read_only(state):
    """Injectors must copy before changing anything, and detectors must not change
    what they're given. Read-only arrays turn either mistake into an error instead of
    a quietly corrupted run."""
    for arr in (state.doc_embeddings, state.query_embeddings):
        if arr is not None:
            arr.setflags(write=False)
    return state


def frozen(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr)
    arr.setflags(write=False)
    return arr


def observe(state, idx, topk_ids, topk_scores, query_embeddings, docs=None) -> Observation:
    """docs: (doc_ids, doc_texts) as tuples, pass them in to avoid re-copying a big corpus."""
    doc_ids, doc_texts = docs or (tuple(state.doc_ids), tuple(state.doc_texts))
    return Observation(
        query_ids=[state.query_ids[i] for i in idx],
        query_embeddings=frozen(query_embeddings),
        topk_ids=[tuple(row) for row in topk_ids],
        topk_scores=frozen(topk_scores),
        query_texts=[state.query_texts[i] for i in idx],
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        doc_embeddings=state.doc_embeddings,
    )


def docs_of(state, clean, clean_docs):
    if state.doc_ids is clean.doc_ids and state.doc_texts is clean.doc_texts:
        return clean_docs
    return tuple(state.doc_ids), tuple(state.doc_texts)


def injector_seed(seed: int, trial: int) -> int:
    # seed + trial would give seed 0 / trial 1 the same damage as seed 1 / trial 0
    return int(np.random.SeedSequence([seed, trial]).generate_state(1)[0])


def run_dataset(ds, retriever, cfg) -> list[ResultRow]:
    cfg = {**DEFAULTS, **cfg}
    k, seed, alpha, min_drop = cfg["k"], cfg["seed"], cfg["alpha"], cfg["min_drop"]
    window_size = cfg["window_size"]
    rid = run_id(cfg)
    failures = cfg.get("failures") or {}
    injectors = lookup(INJECTORS, failures, "injector")
    detectors = lookup(DETECTORS, cfg["detectors"], "detector")

    state = read_only(retriever.embed_corpus(ds))
    clean_docs = (tuple(state.doc_ids), tuple(state.doc_texts))
    clean_ids, clean_scores, clean_q = retriever.search(state, k=k, query_embeddings=state.query_embeddings)
    clean_ndcg = per_query_ndcg(clean_ids, state.query_ids, ds.qrels, k)

    # The reference window the detectors are fit on never overlaps the windows they score.
    order = np.random.default_rng(seed).permutation(len(state.query_ids))
    ref_idx, pool = order[:window_size], order[window_size:]
    if len(pool) < window_size:
        raise SystemExit(f"{ds.name}: {len(order)} queries is too few for two windows of {window_size}")

    reference = observe(
        state, ref_idx, [clean_ids[i] for i in ref_idx], clean_scores[ref_idx], clean_q[ref_idx], clean_docs
    )
    fitted = {}
    for name, cls in detectors.items():
        fitted[name] = cls()
        fitted[name].fit(reference)

    # one injector per failure, so anything it caches (a second model's vectors) is reused across severities
    made = {}
    for name, cls in injectors.items():
        made[name] = cls()
        made[name].attach(retriever)

    conditions = [("clean", 0.0)] + [(f, s) for f, sevs in failures.items() for s in sevs]
    rows = []
    for failure, severity in conditions:
        injector = made.get(failure)
        for trial in range(cfg["trials"]):
            # same windows for every condition, so trial t is comparable across them
            idx = np.random.default_rng([seed, trial]).choice(pool, window_size, replace=False)
            if injector is None:
                damaged = state
                topk_ids = [clean_ids[i] for i in idx]
                topk_scores, q_emb = clean_scores[idx], clean_q[idx]
            else:
                damaged = read_only(injector.apply(state, severity, seed=injector_seed(seed, trial)))
                if len(damaged.query_ids) != len(state.query_ids):
                    raise RuntimeError(f"{failure} changed the number of queries, injectors must keep query positions")
                topk_ids, topk_scores, q_emb = retriever.search(
                    damaged, k=k, query_embeddings=window_queries(retriever, state, damaged, idx)
                )

            # qrels are keyed by the clean query ids. What this means for F4 is still open (#13).
            ndcg = per_query_ndcg(topk_ids, [state.query_ids[i] for i in idx], ds.qrels, k)
            drop, p = paired_bootstrap(clean_ndcg[idx], ndcg, seed=trial)
            window = observe(damaged, idx, topk_ids, topk_scores, q_emb, docs_of(damaged, state, clean_docs))

            for name, det in fitted.items():
                start = time.perf_counter()
                score = det.score(window)
                rows.append(
                    ResultRow(
                        run_id=rid,
                        dataset=ds.name,
                        model=state.config.get("model", ""),
                        model_revision=state.config.get("revision"),
                        failure=failure,
                        expected_harmful=injector.harmful if injector else False,
                        severity=severity,
                        seed=seed,
                        trial=trial,
                        window_size=window_size,
                        detector=name,
                        score=float(score),
                        ndcg_drop=drop,
                        harmful=p < alpha and drop >= min_drop,
                        p_value=p,
                        cost_s=time.perf_counter() - start,
                    )
                )
    return rows


def run(cfg: dict, final: bool = False, loader=data.load, make_retriever=None):
    datasets = check_datasets(cfg["datasets"], final)
    load_plugins()
    check_severities(cfg.get("failures") or {})
    lookup(INJECTORS, cfg.get("failures") or {}, "injector")
    lookup(DETECTORS, cfg["detectors"], "detector")
    make_retriever = make_retriever or (lambda: Retriever(cfg["model"], **cfg.get("retriever", {})))
    rows = []
    for name in datasets:
        rows += run_dataset(loader(name), make_retriever(), cfg)
    return to_frame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", help="parquet path, default results/<config name>.parquet")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    check_datasets(cfg["datasets"], args.final)

    df = run(cfg, final=args.final)
    out = Path(args.out or f"results/{Path(args.config).stem}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
