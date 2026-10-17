"""Experiment runner.

    python -m silent_rag.run --config configs/dev.yaml
    python -m silent_rag.run --config configs/final.yaml --final
"""

from __future__ import annotations

import argparse
import importlib
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
    names = [canonical(d) for d in datasets]
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


def check_severities(failures: dict) -> None:
    bad = {f: s for f, sevs in failures.items() for s in sevs if not 0 < s <= 1}
    if bad:
        raise SystemExit(f"severity is the fraction affected and has to be in (0, 1], got {bad}")


def window_queries(retriever, clean, damaged, idx) -> np.ndarray:
    if damaged.query_embeddings is None:
        return retriever.embed_queries([damaged.query_texts[i] for i in idx])
    if damaged.query_embeddings is clean.query_embeddings and damaged.query_texts != clean.query_texts:
        raise RuntimeError(
            "injector changed query_texts but kept the clean query_embeddings, set them to None"
        )
    return damaged.query_embeddings[idx]


def observe(state, idx, topk_ids, topk_scores, query_embeddings) -> Observation:
    return Observation(
        query_ids=[state.query_ids[i] for i in idx],
        query_embeddings=query_embeddings,
        topk_ids=topk_ids,
        topk_scores=topk_scores,
        query_texts=[state.query_texts[i] for i in idx],
        doc_ids=state.doc_ids,
        doc_texts=state.doc_texts,
        doc_embeddings=state.doc_embeddings,
    )


def run_dataset(ds, retriever, cfg) -> list[ResultRow]:
    k = cfg.get("k", 10)
    window_size = cfg["window_size"]
    seed = cfg.get("seed", 0)
    alpha = cfg.get("alpha", 0.05)
    failures = cfg.get("failures") or {}
    injectors = lookup(INJECTORS, failures, "injector")
    detectors = lookup(DETECTORS, cfg["detectors"], "detector")

    state = retriever.embed_corpus(ds)
    clean_ids, clean_scores, clean_q = retriever.search(state, k=k, query_embeddings=state.query_embeddings)
    clean_ndcg = per_query_ndcg(clean_ids, state.query_ids, ds.qrels, k)

    # The reference window the detectors are fit on never overlaps the windows they score.
    order = np.random.default_rng(seed).permutation(len(state.query_ids))
    ref_idx, pool = order[:window_size], order[window_size:]
    if len(pool) < window_size:
        raise SystemExit(f"{ds.name}: {len(order)} queries is too few for two windows of {window_size}")

    reference = observe(
        state, ref_idx, [clean_ids[i] for i in ref_idx], clean_scores[ref_idx], clean_q[ref_idx]
    )
    fitted = {}
    for name, cls in detectors.items():
        fitted[name] = cls()
        fitted[name].fit(reference)

    conditions = [("clean", 0.0)] + [(f, s) for f, sevs in failures.items() for s in sevs]
    rows = []
    for failure, severity in conditions:
        injector = injectors[failure]() if failure != "clean" else None
        for trial in range(cfg["trials"]):
            # same windows for every condition, so trial t is comparable across them
            idx = np.random.default_rng([seed, trial]).choice(pool, window_size, replace=False)
            if injector is None:
                damaged = state
                topk_ids = [clean_ids[i] for i in idx]
                topk_scores, q_emb = clean_scores[idx], clean_q[idx]
            else:
                damaged = injector.apply(state, severity, seed=seed + trial)
                assert len(damaged.query_ids) == len(state.query_ids), "injectors must keep query positions"
                topk_ids, topk_scores, q_emb = retriever.search(
                    damaged, k=k, query_embeddings=window_queries(retriever, state, damaged, idx)
                )

            # qrels are keyed by the clean query ids. What this means for F4 is still open (#13).
            ndcg = per_query_ndcg(topk_ids, [state.query_ids[i] for i in idx], ds.qrels, k)
            drop, p = paired_bootstrap(clean_ndcg[idx], ndcg, seed=trial)
            window = observe(damaged, idx, topk_ids, topk_scores, q_emb)

            for name, det in fitted.items():
                start = time.perf_counter()
                score = det.score(window)
                rows.append(
                    ResultRow(
                        dataset=ds.name,
                        model=state.config.get("model", ""),
                        failure=failure,
                        expected_harmful=injector.harmful if injector else False,
                        severity=severity,
                        seed=seed,
                        trial=trial,
                        window_size=window_size,
                        detector=name,
                        score=float(score),
                        ndcg_drop=drop,
                        harmful=p < alpha,
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
