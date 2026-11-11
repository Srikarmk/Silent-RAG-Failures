"""Experiment runner.

    python -m silent_rag.run --config configs/dev.yaml
    python -m silent_rag.run --config configs/final.yaml --final
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
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
from silent_rag.pipeline import Retriever, is_commit_hash
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


def check_final_revision(run_cfg: dict, final: bool) -> None:
    """Held-out numbers have to be reproducible, so --final needs the clean model pinned
    to a commit, not whatever the Hub serves that day. Dev runs may leave it unpinned."""
    if final and not is_commit_hash(run_cfg["revision"]):
        raise SystemExit(
            "--final needs the clean model's revision pinned to a 40-character lowercase "
            f"commit hash, got {run_cfg['revision']!r}"
        )


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
    if not failures:
        raise SystemExit(f"failures must list at least one failure (it is missing, null or empty), got {failures!r}")
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
        allowed = INJECTORS[name].severities
        if allowed is not None and any(s not in allowed for s in sevs):
            raise SystemExit(f"{label}: {name} only runs at severities {list(allowed)}, got {sevs}")
        # bind, don't construct: the runner makes exactly one injector per label, later
        try:
            inspect.signature(INJECTORS[name]).bind(**params)
        except TypeError as e:
            raise SystemExit(f"{label}: bad parameters for {name}, given {sorted(params)}: {e}") from None
        for prev, (prev_name, _, prev_params) in parsed.items():
            if prev_name == name and prev_params == params:
                raise SystemExit(f"{label} and {prev} run {name} with the same parameters")
        parsed[label] = (name, sevs, params)
    return parsed


def check_score_depth(cfg: dict) -> None:
    """score_depth is optional; when set it has to reach at least as deep as k."""
    depth, k = cfg.get("score_depth"), cfg.get("k", DEFAULTS["k"])
    if depth is None:
        return
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < k:
        raise SystemExit(f"score_depth must be an integer >= k ({k}), got {depth!r}")


def check_detectors(params: dict[str, dict], cfg: dict) -> None:
    """Each detector's own config checks (Detector.check_config), after score_depth is checked."""
    for name, kwargs in params.items():
        try:
            DETECTORS[name].check_config(dict(kwargs), cfg)
        except ValueError as e:
            raise SystemExit(f"{name}: {e}") from None


def check_injectors(failures: dict, run_cfg: dict) -> None:
    """Each label's injector-specific checks (Injector.check_config), on what failure_params returns."""
    for label, (name, _, params) in failures.items():
        try:
            INJECTORS[name].check_config(dict(params), run_cfg)
        except ValueError as e:
            raise SystemExit(f"{label}: {e}") from None


def default_retriever(cfg: dict) -> Retriever:
    model = cfg.get("model")
    if not isinstance(model, str) or not model.strip():
        raise SystemExit(f"config needs model: <sentence-transformers name>, got {model!r}")
    # an empty `retriever:` block parses as null; treat it like no settings
    return Retriever(model, **(cfg.get("retriever") or {}))


def injector_run_cfg(retriever) -> dict:
    """What Injector.check_config gets: the clean model and revision the run really uses."""
    c = retriever.config
    return {"model": c["model"], "revision": c["revision"]}


def window_queries(retriever, clean, damaged, idx) -> np.ndarray:
    if damaged.query_embeddings is None:
        return retriever.embed_queries([damaged.query_texts[i] for i in idx])
    changed = [i for i in idx if damaged.query_texts[i] != clean.query_texts[i]]
    if changed and np.array_equal(damaged.query_embeddings[changed], clean.query_embeddings[changed]):
        raise RuntimeError(
            "injector changed query_texts but kept the clean query_embeddings, set them to None"
        )
    return damaged.query_embeddings[idx]


def asked_queries(clean, damaged, idx, reference_ids: set[str]) -> list[str]:
    """The query id at each window position after the injector ran.

    An injector that shifts traffic (a query shift, a control that oversamples a topic)
    may put another query of the dataset at a position, with its text. Ground truth then
    pairs the question actually asked with its own clean result. A question from outside
    the dataset has no clean result and no qrels, which is the open reference-standard
    question in #13, so it is refused rather than scored against the wrong question.
    """
    clean_text = dict(zip(clean.query_ids, clean.query_texts))
    asked = [damaged.query_ids[i] for i in idx]
    unknown = sorted({q for q in asked if q not in clean_text})
    if unknown:
        raise RuntimeError(
            f"injector asked {len(unknown)} queries that aren't in the dataset (e.g. {unknown[0]!r}); "
            "they have no qrels, so there is no ground truth for them yet (#13)"
        )
    leaked = sorted(set(asked) & reference_ids)
    if leaked:
        raise RuntimeError(
            f"injector asked {len(leaked)} reference-window queries (e.g. {leaked[0]!r}); "
            "use the ids in Injector.query_pool, the detectors were fit on the others"
        )
    wrong = [i for i in idx if damaged.query_texts[i] != clean_text[damaged.query_ids[i]]]
    if wrong:
        i = wrong[0]
        raise RuntimeError(f"position {i} has query id {damaged.query_ids[i]!r} but another query's text")
    return asked


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


def observe(state, idx, topk_ids, topk_scores, query_embeddings, docs=None, deep_scores=None) -> Observation:
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
        deep_scores=None if deep_scores is None else frozen(deep_scores),
    )


def docs_of(state, clean, clean_docs):
    if state.doc_ids is clean.doc_ids and state.doc_texts is clean.doc_texts and state.parent_ids is clean.parent_ids:
        return clean_docs
    return doc_tuples(state)


def injector_seed(seed: int, trial: int) -> int:
    # seed + trial would give seed 0 / trial 1 the same damage as seed 1 / trial 0
    return int(np.random.SeedSequence([seed, trial]).generate_state(1)[0])


def run_dataset(ds, retriever, cfg, failures: dict, rid: str | None = None) -> list[ResultRow]:
    """failures: what failure_params returns, label -> (injector name, severities, params)."""
    cfg = {**DEFAULTS, **cfg}
    k, seed, alpha, min_drop = cfg["k"], cfg["seed"], cfg["alpha"], cfg["min_drop"]
    window_size = cfg["window_size"]
    rid = rid or run_id(cfg)
    params = detector_params(cfg["detectors"])
    detectors = lookup(DETECTORS, params, "detector")

    state = read_only(retriever.embed_corpus(ds))
    clean_docs = doc_tuples(state)
    clean_ids, clean_scores, clean_q = retriever.search(state, k=k, query_embeddings=state.query_embeddings)

    depth = cfg.get("score_depth")
    if depth is not None and depth > len(set(state.parent_ids or state.doc_ids)):
        raise SystemExit(f"{ds.name}: score_depth {depth} is deeper than the index")

    def deep(st, q_emb):
        # a separate search, so topk_ids and nDCG@k are exactly what they are without score_depth
        # (a deeper search can break score ties at rank k differently)
        return None if depth is None else retriever.search(st, k=depth, query_embeddings=q_emb)[1]

    clean_deep = deep(state, clean_q)
    clean_ndcg = per_query_ndcg(clean_ids, state.query_ids, ds.qrels, k)
    clean_by_id = dict(zip(state.query_ids, clean_ndcg))

    # The reference window the detectors are fit on never overlaps the windows they score.
    order = np.random.default_rng(seed).permutation(len(state.query_ids))
    ref_idx, pool = order[:window_size], order[window_size:]
    ref_ids = {state.query_ids[i] for i in ref_idx}
    if len(pool) < window_size:
        raise SystemExit(f"{ds.name}: {len(order)} queries is too few for two windows of {window_size}")

    reference = observe(
        state, ref_idx, [clean_ids[i] for i in ref_idx], clean_scores[ref_idx], clean_q[ref_idx], clean_docs,
        None if clean_deep is None else clean_deep[ref_idx],
    )
    fitted = {}
    for name, cls in detectors.items():
        fitted[name] = cls(**params[name])
        fitted[name].attach(retriever)
        fitted[name].fit(reference)

    # one injector per label, so anything it caches (a second model's vectors) is reused across severities
    made = {}
    for label, (name, _, kwargs) in failures.items():
        made[label] = INJECTORS[name](**kwargs)
        # the queries an injector may move traffic to: never the reference window's
        made[label].query_pool = [state.query_ids[i] for i in pool]
        made[label].attach(retriever, state)

    conditions = [("clean", 0.0)] + [(f, s) for f, (_, sevs, _) in failures.items() for s in sevs]
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
                deep_scores = None if clean_deep is None else clean_deep[idx]
            else:
                damaged = read_only(injector.apply(state, severity, seed=injector_seed(seed, trial)))
                if len(damaged.query_ids) != len(state.query_ids):
                    raise RuntimeError(f"{failure} changed the number of queries, injectors must keep query positions")
                topk_ids, topk_scores, q_emb = retriever.search(
                    damaged, k=k, query_embeddings=window_queries(retriever, state, damaged, idx)
                )
                deep_scores = deep(damaged, q_emb)

            # ground truth pairs each question actually asked with its own clean result (#13)
            asked = asked_queries(state, damaged, idx, ref_ids)
            ndcg = per_query_ndcg(topk_ids, asked, ds.qrels, k)
            drop, p = paired_bootstrap(np.array([clean_by_id[q] for q in asked]), ndcg, seed=trial)
            window = observe(
                damaged, idx, topk_ids, topk_scores, q_emb, docs_of(damaged, state, clean_docs), deep_scores
            )

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


def run(cfg: dict, final: bool = False, loader=data.load, make_retriever=None, only_seeds=None):
    """only_seeds runs a subset of the config's seeds under the full config's run_id, so
    parts run in separate processes merge into one run (merge_parts)."""
    datasets = check_datasets(cfg["datasets"], final)
    seeds = run_seeds(cfg)
    if only_seeds is not None:
        extra = sorted(set(only_seeds) - set(seeds))
        if extra or not only_seeds:
            raise SystemExit(f"only_seeds must be a non-empty subset of the config's seeds {seeds}, got {only_seeds}")
        seeds = [s for s in seeds if s in set(only_seeds)]
    load_plugins()
    failures = failure_params(cfg.get("failures"))
    params = detector_params(cfg["detectors"])
    for name, cls in lookup(DETECTORS, params, "detector").items():
        try:
            cls(**params[name])  # catch a typo'd parameter now, not after embedding the corpus
        except (TypeError, ValueError) as e:
            raise SystemExit(f"bad parameters for {name}: {e}") from None
    make_retriever = make_retriever or (lambda: default_retriever(cfg))
    # building a Retriever doesn't load the model
    run_cfg = injector_run_cfg(make_retriever())
    check_injectors(failures, run_cfg)
    check_score_depth(cfg)
    check_detectors(params, cfg)
    check_final_revision(run_cfg, final)
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
            rows += run_dataset(ds, retriever, {**per_seed, "seed": seed}, failures, rid=rid)
    return to_frame(rows)


class UniqueKeyLoader(yaml.SafeLoader):
    """yaml.safe_load silently keeps the last of two repeated keys, so a repeated failure
    label would drop the first entry. This refuses them instead."""

    def construct_mapping(self, node, deep=False):
        keys, line = [], None
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if line is None and key in keys:
                line = key_node.start_mark.line + 1  # where the first repeat is written
            keys.append(key)
        repeated = sorted({str(k) for k in keys if keys.count(k) > 1})
        if repeated:
            raise SystemExit(f"line {line}: repeated keys {repeated}")
        return super().construct_mapping(node, deep=deep)


def merge_parts(frames):
    """Rows from runs of the same config on disjoint seeds, as one table. Refuses parts
    from different configs or with a seed in two parts, which would double-count it."""
    import pandas as pd

    frames = list(frames)
    if not frames:
        raise SystemExit("nothing to merge")
    ids = sorted({r for f in frames for r in f["run_id"].unique()})
    if len(ids) != 1:
        raise SystemExit(f"parts come from different configs (run_ids {ids})")
    seen: dict[int, int] = {}
    for i, f in enumerate(frames):
        for s in f["seed"].unique():
            if s in seen:
                raise SystemExit(f"seed {s} is in parts {seen[s]} and {i}")
            seen[s] = i
    return pd.concat(frames, ignore_index=True).sort_values(
        ["seed", "failure", "severity", "trial", "detector"], kind="stable", ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", help="parquet path, default results/<config name>.parquet")
    parser.add_argument("--list", action="store_true", help="print the registered injectors and detectors")
    parser.add_argument("--only-seeds", help="comma-separated subset of the config's seeds, for running parts in parallel")
    parser.add_argument("--merge", nargs="+", metavar="PART", help="merge part parquets into --out instead of running")
    args = parser.parse_args()

    if args.merge:
        import pandas as pd

        if not args.out:
            parser.error("--merge needs --out")
        df = merge_parts(pd.read_parquet(p) for p in args.merge)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"wrote {len(df)} rows from {len(args.merge)} parts to {args.out}")
        return

    if args.list:
        load_plugins()
        print("injectors:", ", ".join(sorted(INJECTORS)) or "none yet")
        print("detectors:", ", ".join(sorted(DETECTORS)) or "none yet")
        return
    if not args.config:
        parser.error("--config is required")

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=UniqueKeyLoader)
    check_datasets(cfg["datasets"], args.final)

    only = None
    if args.only_seeds:
        try:
            only = [int(x) for x in args.only_seeds.split(",")]
        except ValueError:
            parser.error(f"--only-seeds takes comma-separated integers, got {args.only_seeds!r}")
    df = run(cfg, final=args.final, only_seeds=only)
    out = Path(args.out or f"results/{Path(args.config).stem}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
