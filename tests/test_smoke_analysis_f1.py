"""Smoke test (spec section 6): real models, notebook 07's conditions, end to end.

    HF_HUB_OFFLINE=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest -m slow -s

Skips when a model file is not in the local Hugging Face cache; any other problem fails.
HF_HUB_OFFLINE=1 keeps it from contacting the Hub; it must be set before the libraries are imported.
"""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from huggingface_hub import try_to_load_from_cache

from silent_rag import analyze_benchmark_results as abr
from silent_rag import data, pipeline
from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.eval.ground_truth import paired_bootstrap
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.inject.f1_partial_upgrade import PartialUpgrade
from silent_rag.inject.model_check import ModelNotCachedError

ROOT = Path(__file__).resolve().parents[1]
SMOKE, SMOKE_ANALYSIS = ROOT / "experiments/analysis_f1/smoke.yaml", ROOT / "experiments/analysis_f1/smoke.analysis.yaml"
# the project's own embedding cache (conftest points every test at a tmp dir; warm vectors keep this test to minutes)
REAL_CACHE = Path(os.environ.get("SILENT_RAG_CACHE", ROOT / "cache")) / "embeddings"
WINDOW = ["failure", "severity", "seed", "trial"]


def skip_unless_cached(run_config):
    runner.load_plugins()
    retriever_cfg = run_config.get("retriever") or {}
    models = [(run_config["model"], retriever_cfg.get("revision"))]
    for label, (name, _, params) in runner.failure_params(run_config["failures"]).items():
        try:
            INJECTORS[name].check_config(dict(params), {"model": run_config["model"], "revision": retriever_cfg.get("revision")})
        except ModelNotCachedError as e:
            pytest.skip(f"{label}: {e}")
        models.append((params["alt_model"], params["alt_revision"]))
    for repo, revision in models:
        if not any(isinstance(try_to_load_from_cache(repo, f, revision=revision), str)
                   for f in ("model.safetensors", "pytorch_model.bin")):
            pytest.skip(f"{repo}@{revision}: model weights are not in the local Hugging Face cache")


@pytest.mark.slow
def test_smoke_run_and_analysis_meet_the_pass_criteria(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_DIR", REAL_CACHE)
    run_config = abr.load_yaml(SMOKE)
    skip_unless_cached(run_config)

    start = time.perf_counter()
    runs = runner.run(run_config)
    run_seconds = time.perf_counter() - start
    results, out = tmp_path / "smoke.parquet", tmp_path / "out"
    runs.to_parquet(results, index=False)
    start = time.perf_counter()
    abr.main(["--config", str(SMOKE), "--analysis", str(SMOKE_ANALYSIS), "--results", str(results), "--out", str(out)])
    analysis_seconds = time.perf_counter() - start
    print(f"\nsmoke wall-clock: run() {run_seconds:.1f} s, analysis {analysis_seconds:.1f} s")

    # run_id check passed (main stops otherwise) and is recorded
    manifest = pd.read_json(out / "manifest.json", typ="series")
    assert set(runs["run_id"]) == {manifest["run_id"]} == {runner.run_id_from_run_config(run_config)}
    # every table and the manifest written; the reference check (main stops if it fails) among them
    assert {t: e["status"] for t, e in manifest["tables"].items()} == {t: "written" for t in manifest["tables"]}
    assert set(manifest["outputs"]) == {p.name for p in out.glob("*.parquet")}
    reference = pd.read_parquet(out / "check_mean_window_ndcg10_drop_against_reference.parquet")
    assert len(reference) == 6 and (reference["observed"].round(3) == reference["expected"].round(3)).all()
    # the rebuilt windows reproduce every stored p_value and ndcg_drop exactly
    stored = runs.drop_duplicates(WINDOW).set_index(WINDOW)
    rebuilt = pd.read_parquet(out / "window_per_question_ndcg10_differences.parquet")
    groups = rebuilt.groupby(WINDOW, sort=False)
    assert len(groups) == len(stored)
    for key, window in groups:
        drop, p = paired_bootstrap(window["baseline_ndcg10"].to_numpy(), window["modified_ndcg10"].to_numpy(), seed=key[3])
        assert (drop, p) == (stored.loc[key, "ndcg_drop"], stored.loc[key, "p_value"]), key
    # AUROCs in [0, 1] where defined
    for name in ("auroc_statistic_harmful_vs_negative_windows_by_severity",
                 "auroc_statistic_harmful_vs_negative_windows_by_min_ndcg10_drop"):
        auroc = pd.read_parquet(out / f"{name}.parquet")["auroc"].dropna()
        assert len(auroc) and auroc.between(0, 1).all(), name
    # re_embed: exactly 0 on unmodified windows. On an F1 window, exactly 0 when its document sample
    # misses every replaced document and above 0 when it includes one. ReEmbed.rows is a fixed hash of
    # the window's question ids, the detector's seed and the document ids, so the sample is recomputed here.
    re_embed = runs[runs["detector"] == "re_embed"]
    assert (re_embed.loc[re_embed["failure"] == "clean", "score"] == 0).all()
    doc_ids = list(data.load(run_config["datasets"][0]).corpus)  # the runner's row order (cached dataset)
    detector = DETECTORS["re_embed"](**runner.detector_params(run_config["detectors"])["re_embed"])
    window_questions = rebuilt.groupby(WINDOW, sort=False)["question_id"].apply(list)  # in the runner's order
    cases = []
    for row in re_embed[re_embed["failure"] != "clean"].itertuples(index=False):
        key = (row.failure, row.severity, row.seed, row.trial)
        sampled = detector.rows(SimpleNamespace(query_ids=window_questions[key], doc_ids=doc_ids))
        replaced = PartialUpgrade.replaced_document_row_indices(
            len(doc_ids), row.severity, runner.injector_seed(int(row.seed), int(row.trial)))
        overlap = bool(set(sampled.tolist()) & set(replaced.tolist()))
        assert (row.score > 0) if overlap else (row.score == 0), (key, overlap, row.score)
        cases.append({"severity": row.severity, "overlap": overlap})
    print("re_embed F1 windows by severity, sample overlaps the replaced documents:")
    print(pd.DataFrame(cases).value_counts(["severity", "overlap"]).sort_index().to_string())
