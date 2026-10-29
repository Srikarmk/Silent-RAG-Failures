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


def detector_params(entries) -> dict[str, dict]:
    """configs list detectors as names, or {name: {param: value}} for ones that take parameters."""
    params = {}
    for entry in entries:
        if isinstance(entry, str):
            items = [(entry, {})]
        elif isinstance(entry, dict):
            items = list(entry.items())
        else:
            raise SystemExit(f"detector entries are names or {{name: {{param: value}}}}, got {entry!r}")
        for name, kw in items:
            if kw is not None and not isinstance(kw, dict):
                raise SystemExit(f"parameters for {name} must be a mapping, got {kw!r}")
            if name in params:
                raise SystemExit(f"{name} is listed twice, one setting per detector per run")
            params[name] = dict(kw or {})
    return params


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


def failure_params(failures) -> dict[str, tuple[str, list, dict]]:
    """label -> (injector name, severities, constructor params).

    configs give `label: {injector: name, severities: [...], param: value}`, or the
    older `name: [severities]`, where the label is the injector name and there are no params.
    """
    if failures is None:
        return {}
    if not isinstance(failures, dict):
        raise SystemExit(f"failures must be a mapping of label to entry, got {failures!r}")
    parsed, keys = {}, {}
    for label, entry in failures.items():
        if not isinstance(label, str) or not label.strip():
            raise SystemExit(f"failure labels must be non-blank strings, got {label!r}")
        key = label.strip().casefold()
        if key == "clean":
            raise SystemExit(f"{label!r}: 'clean' is the undamaged condition, pick another label")
        if key in keys:
            raise SystemExit(f"{label!r} and {keys[key]!r} differ only in case or spaces, pick distinct labels")
        keys[key] = label
        if isinstance(entry, list):
            name, sevs, params = label, entry, {}
        elif isinstance(entry, dict):
            params = dict(entry)
            name = params.pop("injector", None)
            if not isinstance(name, str) or not name.strip():
                raise SystemExit(f"{label}: needs `injector:` naming a registered injector, got {name!r}")
            if "severities" not in params:
                raise SystemExit(f"{label}: needs `severities: [...]`")
            sevs = params.pop("severities")
        else:
            raise SystemExit(f"{label}: expected [severities] or {{injector: name, severities: [...]}}, got {entry!r}")
        if name not in INJECTORS:
            raise SystemExit(f"{label}: no injector registered as {name!r}, available: {sorted(INJECTORS)}")
        # a label may be its own injector's name, never another one's, or the results table is ambiguous
        others = {n.casefold(): n for n in INJECTORS if n != name}
        if key in others:
            raise SystemExit(f"{label}: is the registered name of {others[key]!r}, pick another label")
        check_severities({label: sevs})
        for prev, (prev_name, _, prev_params) in parsed.items():
            if prev_name == name and prev_params == params:
                raise SystemExit(f"{label} and {prev} run {name} with the same parameters")
        parsed[label] = (name, sevs, params)
    return parsed


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


def doc_tuples(state):
    parents = None if state.parent_ids is None else tuple(state.parent_ids)
    return tuple(state.doc_ids), tuple(state.doc_texts), parents


def observe(state, idx, topk_ids, topk_scores, query_embeddings, docs=None) -> Observation:
    """docs: doc_tuples(state), pass it in to avoid re-copying a big corpus every window."""
    doc_ids, doc_texts, parent_ids = docs or doc_tuples(state)
    return Observation(
        query_ids=[state.query_ids[i] for i in idx],
        query_embeddings=frozen(query_embeddings),
        topk_ids=[tuple(row) for row in topk_ids],
        topk_scores=frozen(topk_scores),
        query_texts=[state.query_texts[i] for i in idx],
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        doc_embeddings=state.doc_embeddings,
        parent_ids=parent_ids,
    )


def docs_of(state, clean, clean_docs):
    if state.doc_ids is clean.doc_ids and state.doc_texts is clean.doc_texts and state.parent_ids is clean.parent_ids:
        return clean_docs
    return doc_tuples(state)


def injector_seed(seed: int, trial: int) -> int:
    # seed + trial would give seed 0 / trial 1 the same damage as seed 1 / trial 0
    return int(np.random.SeedSequence([seed, trial]).generate_state(1)[0])


def run_dataset(ds, retriever, cfg, rid: str | None = None) -> list[ResultRow]:
    cfg = {**DEFAULTS, **cfg}
    k, seed, alpha, min_drop = cfg["k"], cfg["seed"], cfg["alpha"], cfg["min_drop"]
    window_size = cfg["window_size"]
    rid = rid or run_id(cfg)
    failures = cfg.get("failures") or {}
    injectors = lookup(INJECTORS, failures, "injector")
    params = detector_params(cfg["detectors"])
    detectors = lookup(DETECTORS, params, "detector")

    state = read_only(retriever.embed_corpus(ds))
    clean_docs = doc_tuples(state)
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
        fitted[name] = cls(**params[name])
        fitted[name].attach(retriever)
        fitted[name].fit(reference)

    # one injector per failure, so anything it caches (a second model's vectors) is reused across severities
    made = {}
    for name, cls in injectors.items():
        made[name] = cls()
        made[name].attach(retriever, state)

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


def run_seeds(cfg: dict) -> list[int]:
    """`seeds: [0, 1, 2]` repeats the whole run with independent windows, reference
    window and damage, so results have real replicates to bootstrap over (#17)."""
    if "seeds" not in cfg:
        return [cfg.get("seed", DEFAULTS["seed"])]
    if "seed" in cfg:
        raise SystemExit("give either seed or seeds, not both")
    seeds = cfg["seeds"]
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise SystemExit(f"seeds must be a non-empty list of integers, got {seeds!r}")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"seeds has duplicates: {seeds}")
    return seeds


def run(cfg: dict, final: bool = False, loader=data.load, make_retriever=None):
    datasets = check_datasets(cfg["datasets"], final)
    seeds = run_seeds(cfg)
    load_plugins()
    check_severities(cfg.get("failures") or {})
    lookup(INJECTORS, cfg.get("failures") or {}, "injector")
    params = detector_params(cfg["detectors"])
    for name, cls in lookup(DETECTORS, params, "detector").items():
        try:
            cls(**params[name])  # catch a typo'd parameter now, not after embedding the corpus
        except TypeError as e:
            raise SystemExit(f"bad parameters for {name}: {e}") from None
    make_retriever = make_retriever or (lambda: Retriever(cfg["model"], **cfg.get("retriever", {})))
    # one run_id for all seeds, so the replicates can be grouped together
    hashed = {**DEFAULTS, **cfg}
    if "seeds" in cfg:
        hashed.pop("seed")
    rid = run_id(hashed)
    per_seed = {key: v for key, v in cfg.items() if key != "seeds"}
    rows = []
    for name in datasets:
        ds, retriever = loader(name), make_retriever()
        for seed in seeds:
            rows += run_dataset(ds, retriever, {**per_seed, "seed": seed}, rid=rid)
    return to_frame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", help="parquet path, default results/<config name>.parquet")
    parser.add_argument("--list", action="store_true", help="print the registered injectors and detectors")
    args = parser.parse_args()

    if args.list:
        load_plugins()
        print("injectors:", ", ".join(sorted(INJECTORS)) or "none yet")
        print("detectors:", ", ".join(sorted(DETECTORS)) or "none yet")
        return
    if not args.config:
        parser.error("--config is required")

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
