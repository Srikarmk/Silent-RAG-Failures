import pandas as pd
import pytest
import yaml

from silent_rag import analyze_benchmark_results as abr
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
    status = abr.analyze(RUN, analysis, runs, out, **FAKES)
    assert set(status.values()) == {"written"}, status
    expected = {f"{n}.parquet" for n in status if n != "window_mean_question_vulnerability_by_label"}
    expected |= {f"window_mean_question_vulnerability_by_label_{p}.parquet" for p in ("per_window", "summary")}
    assert {p.name for p in out.iterdir()} == expected


def test_only_skip_and_the_expensive_inputs_computed_once(toy, monkeypatch):
    runs, analysis, out = toy
    calls = []
    for name in ("per_question_ndcg10_by_failure_severity_seed", "window_per_question_ndcg10_differences"):
        original = getattr(abr, name)
        monkeypatch.setattr(abr, name, lambda *a, _f=original, _n=name, **kw: calls.append(_n) or _f(*a, **kw))
    status = abr.analyze(RUN, analysis, runs, out, skip=["check_mean_window_ndcg10_drop_against_reference"], **FAKES)
    assert "check_mean_window_ndcg10_drop_against_reference" not in status and len(status) == 15
    assert sorted(calls) == ["per_question_ndcg10_by_failure_severity_seed", "window_per_question_ndcg10_differences"]
    only = abr.analyze(RUN, analysis, runs, out / "only", only=["median_seconds_per_statistic_evaluation"], **FAKES)
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
    assert abr.analyze(RUN, without, runs, out, only=[name], **FAKES) == {
        name: "skipped: the analysis config has no window_bootstrap"}

    def broken(runs):
        raise ValueError("broken table")

    monkeypatch.setattr(abr.wa, "median_seconds_per_statistic_evaluation", broken)
    with pytest.raises(ValueError, match="broken table"):
        abr.analyze(RUN, analysis, runs, out, only=["median_seconds_per_statistic_evaluation"], **FAKES)


def test_run_id_mismatch_stops(toy):
    runs, analysis, out = toy
    files = {name: out.parent / f"{name}.yaml" for name in ("run", "analysis")}
    files["run"].write_text(yaml.safe_dump({**RUN, "trials": 3}))  # another run config, another run_id
    files["analysis"].write_text(yaml.safe_dump(analysis))
    runs.to_parquet(out.parent / "runs.parquet")
    with pytest.raises(SystemExit, match="results have run_id"):
        abr.main(["--config", str(files["run"]), "--analysis", str(files["analysis"]),
                  "--results", str(out.parent / "runs.parquet"), "--out", str(out)])
