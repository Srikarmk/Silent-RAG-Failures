import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.eval.ground_truth import bootstrap_ci, paired_bootstrap
from silent_rag.eval.window_questions import (
    window_bootstrap_ci_vs_mean_ndcg10_drop,
    window_per_question_ndcg10_differences,
    window_question_ids,
)
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.pipeline import Retriever
from test_pipeline import HashEncoder, MeanTop1, ScrambleDocs, toy_dataset

RUN_CONFIG = {"datasets": ["nfcorpus"], "window_size": 10, "trials": 3, "seeds": [0, 1],
              "failures": {ScrambleDocs.name: [0.3, 0.9]}, "detectors": [MeanTop1.name]}
FAKES = {"loader": lambda name: toy_dataset(), "make_retriever": lambda: Retriever("clean/model", encoder=HashEncoder())}


@pytest.fixture
def toy_run(monkeypatch):
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    monkeypatch.setitem(DETECTORS, MeanTop1.name, MeanTop1)
    runs = runner.run(RUN_CONFIG, **FAKES)
    return runs, window_per_question_ndcg10_differences(RUN_CONFIG, runs, **FAKES)


def test_rebuilt_windows_reproduce_the_stored_p_value_and_drop_exactly(toy_run):
    runs, rebuilt = toy_run
    stored = runs.drop_duplicates(["failure", "severity", "seed", "trial"]).set_index(["failure", "severity", "seed", "trial"])
    groups = rebuilt.groupby(["failure", "severity", "seed", "trial"], sort=False)
    assert len(groups) == len(stored) == (1 + 2) * 2 * 3  # clean + two severities, two seeds, three trials
    for key, window in groups:
        drop, p = paired_bootstrap(window["baseline_ndcg10"].to_numpy(), window["modified_ndcg10"].to_numpy(), seed=key[3])
        assert (drop, p) == (stored.loc[key, "ndcg_drop"], stored.loc[key, "p_value"]), key
    assert (rebuilt.loc[rebuilt.failure == "clean", "baseline_ndcg10"]
            == rebuilt.loc[rebuilt.failure == "clean", "modified_ndcg10"]).all()


def test_window_ids_match_window_question_indices(toy_run):
    _, rebuilt = toy_run
    question_ids = list(toy_dataset().queries)
    for seed, trial in [(0, 0), (1, 2)]:
        reference, window = window_question_ids(RUN_CONFIG, seed, trial, loader=FAKES["loader"])
        ref_idx, win_idx = runner.window_question_indices(len(question_ids), 10, seed, trial)
        assert (reference, window) == ([question_ids[i] for i in ref_idx], [question_ids[i] for i in win_idx])
        got = rebuilt[(rebuilt.failure == ScrambleDocs.name) & (rebuilt.severity == 0.9)
                      & (rebuilt.seed == seed) & (rebuilt.trial == trial)]
        assert got["question_id"].tolist() == window  # in the window's index order


WINDOW_BOOT = {"window_bootstrap": {"resamples": 2000, "seed": 0}}


def test_window_ci_mean_is_the_stored_drop_and_ci_is_bootstrap_ci(toy_run):
    runs, rebuilt = toy_run
    keys = ["failure", "severity", "seed", "trial"]
    stored = runs.drop_duplicates(keys).set_index(keys)
    got = window_bootstrap_ci_vs_mean_ndcg10_drop(rebuilt, WINDOW_BOOT)
    assert len(got) == len(stored) and (got["n_questions"] == RUN_CONFIG["window_size"]).all()
    for row in got.itertuples(index=False):
        key = (row.failure, row.severity, row.seed, row.trial)
        assert row.mean_ndcg10_drop == stored.loc[key, "ndcg_drop"], key  # exact, the runner's mean
        window = rebuilt.set_index(keys).loc[key]
        drops = window["baseline_ndcg10"].to_numpy() - window["modified_ndcg10"].to_numpy()
        assert (row.ci_low, row.ci_high) == bootstrap_ci(drops, n_boot=2000, seed=0)[1:]
        assert row.ci_low <= row.mean_ndcg10_drop <= row.ci_high


def test_window_ci_is_zero_for_unmodified_windows(toy_run):
    got = window_bootstrap_ci_vs_mean_ndcg10_drop(toy_run[1], WINDOW_BOOT)
    clean = got[got.failure == "clean"]
    assert len(clean) == 6 and (clean[["mean_ndcg10_drop", "ci_low", "ci_high"]] == 0.0).all(axis=None)


class MeanGap(MeanTop1):
    """A second statistic, so the results table has two rows per window."""
    name = "test_mean_gap"

    def score(self, window):
        return float((window.topk_scores[:, 0] - window.topk_scores[:, 1]).mean())


def test_two_statistics_give_one_set_of_rows_per_window(monkeypatch):
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    monkeypatch.setitem(DETECTORS, MeanTop1.name, MeanTop1)
    monkeypatch.setitem(DETECTORS, MeanGap.name, MeanGap)
    config = {**RUN_CONFIG, "detectors": [MeanTop1.name, MeanGap.name]}
    runs = runner.run(config, **FAKES)
    keys = ["failure", "severity", "seed", "trial"]
    stored = runs.drop_duplicates(keys).set_index(keys)
    assert len(runs) == 2 * len(stored)  # two statistics per window
    rebuilt = window_per_question_ndcg10_differences(config, runs, **FAKES)
    assert len(rebuilt) == len(stored) * config["window_size"]
    assert not rebuilt.duplicated([*keys, "question_id"]).any()
    for key, window in rebuilt.groupby(keys, sort=False):
        drop, p = paired_bootstrap(window["baseline_ndcg10"].to_numpy(), window["modified_ndcg10"].to_numpy(), seed=key[3])
        assert (drop, p) == (stored.loc[key, "ndcg_drop"], stored.loc[key, "p_value"]), key


@pytest.mark.parametrize("run_ids, match", [(["other"], "run_id \\['other'\\]"), (["a", "b"], "run_id \\['a', 'b'\\]")])
def test_mismatched_or_mixed_run_ids_raise(toy_run, run_ids, match):
    runs, _ = toy_run
    runs = runs.assign(run_id=[run_ids[i % len(run_ids)] for i in range(len(runs))])
    with pytest.raises(ValueError, match=match):
        window_per_question_ndcg10_differences(RUN_CONFIG, runs, **FAKES)
