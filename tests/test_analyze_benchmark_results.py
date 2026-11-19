import hashlib
import json

import pandas as pd
import pytest
import yaml

from silent_rag import analyze_benchmark_results as abr
from silent_rag import manifest as manifest_module
from silent_rag import run as runner
from silent_rag.inject.f1_partial_upgrade import PartialUpgrade
from silent_rag.pipeline import Retriever
from test_all_test_questions import with_validation
from test_inject_f1 import ShiftedEncoder
from test_pipeline import HashEncoder

RUN = {"datasets": ["nfcorpus"], "model": "clean/model", "window_size": 10, "trials": 2, "seeds": [0, 1],
       "detectors": ["ks_top1", "psi_top1"], "failures": {"F1_shifted": {
           "injector": "F1_partial_upgrade", "severities": [0.5, 0.9], "alt_model": "alt/model", "alt_revision": "a" * 40}}}
BOOT = {"resamples": 50, "seed": 0}
FAKES = {"loader": with_validation(), "make_retriever": lambda: Retriever("clean/model", encoder=HashEncoder())}


@pytest.fixture
def toy(monkeypatch, tmp_path):
    class Shifted(Retriever):
        def __init__(self, model, **kw):
            super().__init__(model, encoder=ShiftedEncoder(), **kw)

    monkeypatch.setattr(PartialUpgrade, "retriever_cls", Shifted)
    monkeypatch.setattr(PartialUpgrade, "model_checker", staticmethod(lambda *a: None))  # offline
    runs = runner.run(RUN, **FAKES)
    windows = runs.drop_duplicates(["severity", "seed", "trial"])
    means = windows[windows.failure == "F1_shifted"].groupby("severity").ndcg_drop.mean().round(3).to_dict()
    analysis = {"min_mean_ndcg10_drops_for_harmful_share": [0.0, 0.05], "question_bootstrap": BOOT,
                "min_mean_ndcg10_drop_grid_for_auroc": {"start": 0.0, "stop": 0.02, "step": 0.01},
                "window_bootstrap": BOOT, "seed_cluster_bootstrap": BOOT,
                "question_vulnerability": {"top_fractions": [0.25], "permutations": 50, "seed": 0},
                "injector_specific": {"F1_partial_upgrade": {"reference_mean_window_ndcg10_drop": {
                    "seeds": [0, 1], "values": {"F1_shifted": means}}}}}
    return runs, analysis, tmp_path / "out"


def test_every_table_is_written(toy):
    runs, analysis, out = toy
    status, seconds = abr.analyze(RUN, analysis, runs, out, **FAKES)
    assert set(status.values()) == {"written"}, status
    assert set(seconds) == set(status) and all(s > 0 for s in seconds.values())
    expected = {f"{n}.parquet" for n in status if n != "window_mean_question_vulnerability_by_label"}
    expected |= {f"window_mean_question_vulnerability_by_label_{p}.parquet" for p in ("per_window", "summary")}
    assert {p.name for p in out.iterdir()} == expected


def test_only_skip_and_the_expensive_inputs_computed_once(toy, monkeypatch):
    runs, analysis, out = toy
    calls = []
    for name in ("per_question_ndcg10_by_failure_severity_seed", "window_per_question_ndcg10_differences"):
        original = getattr(abr, name)
        monkeypatch.setattr(abr, name, lambda *a, _f=original, _n=name, **kw: calls.append(_n) or _f(*a, **kw))
    status, _ = abr.analyze(RUN, analysis, runs, out, skip=["check_mean_window_ndcg10_drop_against_reference"], **FAKES)
    assert "check_mean_window_ndcg10_drop_against_reference" not in status and len(status) == 15
    assert sorted(calls) == ["per_question_ndcg10_by_failure_severity_seed", "window_per_question_ndcg10_differences"]
    only, _ = abr.analyze(RUN, analysis, runs, out / "only", only=["median_seconds_per_statistic_evaluation"], **FAKES)
    assert list(only) == ["median_seconds_per_statistic_evaluation"]


def test_auroc_by_severity_runs_at_every_minimum_drop_in_the_list(toy):
    runs, analysis, out = toy
    only = ["auroc_statistic_harmful_vs_negative_windows_by_severity"]
    abr.analyze(RUN, analysis, runs, out, only=only, **FAKES)
    got = pd.read_parquet(out / f"{only[0]}.parquet")
    assert sorted(got["min_mean_ndcg10_drop"].unique()) == [0.0, 0.05]  # the analysis config's list


def test_missing_setting_skips_its_table_but_an_error_inside_a_table_stops_the_run(toy, monkeypatch):
    runs, analysis, out = toy
    without = {k: v for k, v in analysis.items() if k != "window_bootstrap"}
    name = "window_bootstrap_ci_vs_mean_ndcg10_drop"
    assert abr.analyze(RUN, without, runs, out, only=[name], **FAKES) == (
        {name: "skipped: the analysis config has no window_bootstrap"}, {name: None})

    def broken(runs):
        raise ValueError("broken table")

    monkeypatch.setattr(abr.wa, "median_seconds_per_statistic_evaluation", broken)
    with pytest.raises(ValueError, match="broken table"):
        abr.analyze(RUN, analysis, runs, out, only=["median_seconds_per_statistic_evaluation"], **FAKES)


def cli(out, run_config, analysis, runs):
    """Write the three input files next to out; return main()'s arguments and the paths."""
    paths = {"run_config": out.parent / "run.yaml", "analysis_config": out.parent / "analysis.yaml",
             "results": out.parent / "runs.parquet"}
    paths["run_config"].write_text(yaml.safe_dump(run_config))
    paths["analysis_config"].write_text(yaml.safe_dump(analysis))
    runs.to_parquet(paths["results"])
    return ["--config", str(paths["run_config"]), "--analysis", str(paths["analysis_config"]),
            "--results", str(paths["results"]), "--out", str(out)], paths


def test_run_id_mismatch_stops(toy):
    runs, analysis, out = toy
    argv, _ = cli(out, {**RUN, "trials": 3}, analysis, runs)  # another run config, another run_id
    with pytest.raises(SystemExit, match="results have run_id"):
        abr.main(argv, **FAKES)


def test_main_writes_a_manifest_of_every_table_and_input(toy):
    runs, analysis, out = toy
    argv, paths = cli(out, RUN, {k: v for k, v in analysis.items() if k != "window_bootstrap"}, runs)
    abr.main(argv, **FAKES)
    manifest = json.loads((out / "manifest.json").read_text())
    tables = sorted(p for p in out.iterdir() if p.suffix == ".parquet")
    assert len(tables) == 16  # 15 tables written, one of them as two files; window_bootstrap is skipped
    assert manifest["outputs"] == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tables}
    assert {role: entry["sha256"] for role, entry in manifest["inputs"].items()} == {
        role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in paths.items()}
    skipped = manifest["tables"]["window_bootstrap_ci_vs_mean_ndcg10_drop"]
    assert skipped == {"status": "skipped: the analysis config has no window_bootstrap", "seconds": None}
    written = [t for t, entry in manifest["tables"].items() if entry["status"] == "written"]
    assert len(written) == 15 and all(manifest["tables"][t]["seconds"] > 0 for t in written)
    assert manifest["run_id"] == runner.run_id_from_run_config(RUN)


def test_main_reads_the_git_state_before_the_run_and_again_when_writing(toy, monkeypatch):
    runs, analysis, out = toy
    argv, _ = cli(out, RUN, analysis, runs)
    calls = []
    def state(commit, paths):
        return {"commit": commit, "uncommitted_changes": bool(paths), "uncommitted_paths": paths, "note": None}
    def logged(name, function):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return function(*args, **kwargs)
        return wrapper
    monkeypatch.setattr(abr, "git_state", logged("start", lambda repo: state("c" * 40, ["x.py"])))
    monkeypatch.setattr(manifest_module, "git_state", logged("end", lambda repo: state("d" * 40, ["x.py", "late.txt"])))
    monkeypatch.setattr(abr, "load_yaml", logged("yaml", abr.load_yaml))
    monkeypatch.setattr(abr, "analyze", logged("analyze", abr.analyze))
    abr.main(argv, **FAKES)
    assert calls == ["start", "yaml", "yaml", "analyze", "end"]  # start before anything, end after the tables
    git = json.loads((out / "manifest.json").read_text())["git"]
    assert git == {"commit": "c" * 40, "uncommitted_changes": True, "uncommitted_paths": ["x.py"], "note": None,
                   "end_commit": "d" * 40, "uncommitted_paths_at_end": ["x.py", "late.txt"],
                   "commit_changed_during_run": True}
