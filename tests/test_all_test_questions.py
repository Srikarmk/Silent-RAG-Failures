import numpy as np
import pandas as pd
import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.detect.base import Detector
from silent_rag.eval.all_test_questions import (
    mean_ndcg10_drop_all_test_questions_by_severity,
    per_question_ndcg10_by_failure_severity_seed,
    statistic_all_test_questions_vs_validation_reference_by_severity,
)
from silent_rag.eval.ground_truth import bootstrap_ci
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.inject.f1_partial_upgrade import PartialUpgrade
from silent_rag.pipeline import Retriever
from silent_rag.types import Dataset
from test_inject_f1 import ShiftedEncoder
from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset

F1_SEVERITIES, SEEDS = [0.25, 0.5], [0, 3]
RUN_CONFIG = {
    "datasets": ["nfcorpus"], "model": "clean/model", "seeds": SEEDS,
    "failures": {"F1_shifted": {"injector": "F1_partial_upgrade", "severities": F1_SEVERITIES,
                                "alt_model": "alt/model", "alt_revision": "a" * 40},
                 ScrambleDocs.name: [0.5]},
}


@pytest.fixture
def table(monkeypatch):
    class Shifted(Retriever):  # every replacement vector differs from the baseline's
        def __init__(self, model, **kw):
            super().__init__(model, encoder=ShiftedEncoder(), **kw)

    monkeypatch.setattr(PartialUpgrade, "retriever_cls", Shifted)
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    return per_question_ndcg10_by_failure_severity_seed(
        RUN_CONFIG, {}, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("clean/model", encoder=HashEncoder()))


def test_one_row_per_question_label_severity_seed(table):
    n_questions = len(toy_dataset().queries)
    assert len(table) == n_questions * (len(F1_SEVERITIES) + 1) * len(SEEDS)
    assert not table.duplicated(["question_id", "failure", "severity", "seed"]).any()
    assert table.groupby("failure")["injector"].first().to_dict() == {
        "F1_shifted": "F1_partial_upgrade", ScrambleDocs.name: ScrambleDocs.name}


def test_baseline_ndcg_is_the_same_for_every_severity_and_seed_of_a_question(table):
    assert (table.groupby("question_id")["baseline_ndcg10"].nunique() == 1).all()
    assert (table["n_relevant"] == 1).all()  # toy dataset: q_i has one relevant document, d_i


@pytest.mark.parametrize("severity", F1_SEVERITIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_relevant_modified_counts_use_the_raw_seed(table, severity, seed):
    got = table[(table.failure == "F1_shifted") & (table.severity == severity) & (table.seed == seed)]
    rows = set(PartialUpgrade.replaced_document_row_indices(40, severity, seed).tolist())
    expected = [int(int(q[1:]) in rows) for q in got["question_id"]]  # q_i's relevant document is row i
    assert got["n_relevant_modified"].tolist() == expected
    # the runner's per-window seed would pick other rows, so this test tells the two apart
    window_rows = set(PartialUpgrade.replaced_document_row_indices(40, severity, runner.injector_seed(seed, 0)).tolist())
    assert window_rows != rows


def test_injector_without_the_hook_gives_null_not_zero(table):
    scramble = table[table.failure == ScrambleDocs.name]["n_relevant_modified"]
    assert scramble.isna().all() and str(scramble.dtype) == "Int64"


class ScaledEncoder(HashEncoder):
    """Replacement vectors 0.3 x the baseline's: a replaced d_i still scores above 0 for q_i, but
    below documents sharing a hashed word with q_i, so its rank is informative and mostly untied.
    (ShiftedEncoder's replaced documents all score 0 and tie with ~38 others.)"""

    def encode(self, texts, **kw):
        return 0.3 * super().encode(texts, **kw)


def test_modified_ndcg_of_replaced_questions_by_hand(monkeypatch):
    """Recompute modified nDCG@10 with plain numpy (no FAISS, no per_query_ndcg) for every question
    whose relevant document was replaced and whose rank does not depend on the search's tie order."""
    class Scaled(Retriever):
        def __init__(self, model, **kw):
            super().__init__(model, encoder=ScaledEncoder(), **kw)

    monkeypatch.setattr(PartialUpgrade, "retriever_cls", Scaled)
    config = {**RUN_CONFIG, "seeds": [0], "failures": {"F1_scaled": {**RUN_CONFIG["failures"]["F1_shifted"],
                                                                     "alt_model": "scaled/model", "severities": [0.5]}}}
    got = per_question_ndcg10_by_failure_severity_seed(
        config, {}, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("clean/model", encoder=HashEncoder()))
    ds = toy_dataset()
    texts = list(ds.corpus.values())
    vectors = HashEncoder().encode(texts)
    rows = PartialUpgrade.replaced_document_row_indices(len(texts), 0.5, 0)
    vectors[rows] = ScaledEncoder().encode([texts[r] for r in rows])
    checked = []
    for i in sorted(rows.tolist()):  # q_i's one relevant document is row i, and it was replaced
        scores = vectors @ HashEncoder().encode([ds.queries[f"q{i}"]])[0]
        above, tied = int((scores > scores[i]).sum()), int((scores == scores[i]).sum()) - 1
        if tied and above < 10:
            continue  # d_i's position would depend on the search's tie order
        expected = 1 / np.log2(above + 2) if above < 10 else 0.0  # one relevant document of grade 1: IDCG = 1
        assert got.loc[got.question_id == f"q{i}", "modified_ndcg10"].item() == pytest.approx(expected), i
        checked.append(expected)
    assert min(checked) < 1  # at least one replaced question ranks its document below the top


def per_question(drops):
    """drops: {question: [drop per seed]}; baseline 0.5, so modified = 0.5 - drop."""
    return pd.DataFrame([{"question_id": q, "failure": "F1", "injector": "F1_partial_upgrade", "severity": 0.5,
                          "seed": s, "baseline_ndcg10": 0.5, "modified_ndcg10": 0.5 - d}
                         for q, ds in drops.items() for s, d in enumerate(ds)])


BOOT = {"question_bootstrap": {"resamples": 2000, "seed": 0}}


def label_rows(table):
    """The failure x severity rows of table 9, without its clean row."""
    return table[table["failure"].ne("clean")].reset_index(drop=True)


def test_mean_drop_is_a_hand_calculation_with_counts():
    got = label_rows(mean_ndcg10_drop_all_test_questions_by_severity(
        per_question({"a": [0.1, 0.3], "b": [0.0, 0.2]}), BOOT)).iloc[0]
    assert got["mean_ndcg10_drop"] == pytest.approx((0.2 + 0.1) / 2)  # question means 0.2 and 0.1
    assert (got["n_questions"], got["n_seeds"], got["injector"]) == (2, 2, "F1_partial_upgrade")


def test_seeds_are_averaged_before_the_question_bootstrap():
    # questions differ (0.0 to 0.3); seeds of a question barely do. Five seeds per question.
    drops = {f"q{i}": [0.1 * i + 0.001 * s for s in range(5)] for i in range(4)}
    got = label_rows(mean_ndcg10_drop_all_test_questions_by_severity(per_question(drops), BOOT)).iloc[0]
    question_means = np.array([np.mean(d) for d in drops.values()])
    assert (got["ci_low"], got["ci_high"]) == pytest.approx(bootstrap_ci(question_means, n_boot=2000, seed=0)[1:])
    pooled = bootstrap_ci(np.concatenate(list(drops.values())), n_boot=2000, seed=0)  # 20 "independent" values
    assert got["ci_high"] - got["ci_low"] > pooled[2] - pooled[1]


def test_baseline_ndcg10_is_the_mean_over_questions_and_constant_per_run():
    # baselines a 0.8, b 0.2, c 0.5 (same in every seed and condition); two failures x two severities, two seeds
    baselines = {"a": 0.8, "b": 0.2, "c": 0.5}
    rows = pd.DataFrame([{"question_id": q, "failure": f, "injector": f, "severity": sev, "seed": s,
                          "baseline_ndcg10": b, "modified_ndcg10": b - 0.1 * sev}
                         for f in ("F1", "F2") for sev in (0.1, 0.9) for s in (0, 1) for q, b in baselines.items()])
    got = mean_ndcg10_drop_all_test_questions_by_severity(rows, BOOT)
    assert got["baseline_ndcg10"].tolist() == pytest.approx([(0.8 + 0.2 + 0.5) / 3] * 5)  # clean + four label rows
    assert list(got.columns) == ["failure", "injector", "severity", "mean_ndcg10_drop", "ci_low", "ci_high",
                                 "n_questions", "n_seeds", "baseline_ndcg10"]  # existing columns first, in order
    assert label_rows(got)["mean_ndcg10_drop"].tolist() == pytest.approx([0.01, 0.09, 0.01, 0.09])


def test_clean_row_comes_first_with_values_written_and_label_rows_unchanged():
    baselines = {"a": 0.8, "b": 0.2, "c": 0.5}
    rows = pd.DataFrame([{"question_id": q, "failure": f, "injector": f, "severity": sev, "seed": s,
                          "baseline_ndcg10": b, "modified_ndcg10": b - 0.1 * sev - 0.01 * s}
                         for f in ("F1", "F2") for sev in (0.1, 0.9) for s in (0, 1) for q, b in baselines.items()])
    got = mean_ndcg10_drop_all_test_questions_by_severity(rows, BOOT)
    assert (got["failure"] == "clean").sum() == 1 and got.loc[0, "failure"] == "clean"
    clean = got.iloc[0]
    assert pd.isna(clean["injector"]) and clean["severity"] == 0.0
    assert (clean["mean_ndcg10_drop"], clean["ci_low"], clean["ci_high"]) == (0.0, 0.0, 0.0)  # exact, not estimated
    assert (clean["n_questions"], clean["n_seeds"]) == (3, 0)
    assert clean["baseline_ndcg10"] == got.loc[1, "baseline_ndcg10"]  # the same value as the label rows, exactly
    again = label_rows(mean_ndcg10_drop_all_test_questions_by_severity(rows, BOOT))
    pd.testing.assert_frame_equal(got.iloc[1:].reset_index(drop=True), again)
    assert list(again[["failure", "severity"]].itertuples(index=False, name=None)) == [
        ("F1", 0.1), ("F1", 0.9), ("F2", 0.1), ("F2", 0.9)]  # label rows in their usual order


def test_empty_input_gives_an_empty_table_without_a_clean_row():
    empty = per_question({}).reindex(columns=["question_id", "failure", "injector", "severity", "seed",
                                              "baseline_ndcg10", "modified_ndcg10"])
    assert mean_ndcg10_drop_all_test_questions_by_severity(empty, BOOT).empty


def test_unequal_seed_counts_are_an_error():
    with pytest.raises(ValueError, match="different numbers of seeds"):
        mean_ndcg10_drop_all_test_questions_by_severity(per_question({"a": [0.1, 0.3], "b": [0.0]}), BOOT)


class Recording(Detector):
    """Records construction, attach, fit and score; scores a window by its mean top-1 score x scale."""
    name = "test_recording"
    events = []

    def __init__(self, scale=1.0):
        self.scale = scale
        Recording.events.append(("init", scale))

    def attach(self, retriever):
        Recording.events.append(("attach", None))

    def fit(self, reference):
        Recording.events.append(("fit", list(reference.query_ids)))

    def score(self, window):
        value = self.scale * float(window.topk_scores[:, 0].mean())
        Recording.events.append(("score", (len(window.query_ids), value)))
        return value


def with_validation(validation_corpus_size=40, shared_questions=()):
    """Test split: the toy dataset. Validation: 20 other questions over a corpus of the given size,
    plus any test question ids in shared_questions."""
    def loader(name, split="test"):
        if split == "test":
            return toy_dataset()
        corpus = toy_dataset(validation_corpus_size).corpus
        queries = {f"v{i}": f"word{i} topic{i}" for i in range(20)} | {q: "shared" for q in shared_questions}
        return Dataset("nfcorpus", corpus, queries, {q: {"d0": 1} for q in queries})
    return loader


@pytest.fixture
def statistic_table(monkeypatch):
    class Shifted(Retriever):
        def __init__(self, model, **kw):
            super().__init__(model, encoder=ShiftedEncoder(), **kw)

    monkeypatch.setattr(PartialUpgrade, "retriever_cls", Shifted)
    monkeypatch.setitem(DETECTORS, Recording.name, Recording)
    monkeypatch.setattr(Recording, "events", [])
    config = {**RUN_CONFIG, "failures": {"F1_shifted": {**RUN_CONFIG["failures"]["F1_shifted"], "severities": [0.5]}},
              "detectors": [{Recording.name: {"scale": 2.0}}]}
    table = statistic_all_test_questions_vs_validation_reference_by_severity(
        config, {}, loader=with_validation(), make_retriever=lambda: Retriever("clean/model", encoder=HashEncoder()))
    return table, Recording.events


def test_statistic_is_built_from_config_attached_then_fitted_on_validation_only(statistic_table):
    _, events = statistic_table
    assert [e for e, _ in events[:3]] == ["init", "attach", "fit"] and events[0][1] == 2.0
    assert events[2][1] == [f"v{i}" for i in range(20)]  # validation questions only
    scores = [value for e, value in events if e == "score"]
    assert all(n == 40 for n, _ in scores)  # every scored window is all 40 test questions


def test_clean_row_and_range_over_seeds(statistic_table):
    table, events = statistic_table
    scores = [value[1] for e, value in events if e == "score"]  # clean first, then seeds 0 and 3
    clean = table[table.failure == "clean"].iloc[0]
    assert (clean["severity"], clean["n_seeds"], clean["mean"]) == (0.0, 0, pytest.approx(scores[0]))
    assert pd.isna(clean["injector"])
    f1 = table[table.failure == "F1_shifted"].iloc[0]
    assert (f1["mean"], f1["min"], f1["max"]) == pytest.approx((np.mean(scores[1:]), min(scores[1:]), max(scores[1:])))
    assert (f1["n_seeds"], f1["injector"], f1["statistic"]) == (2, "F1_partial_upgrade", Recording.name)
    assert len(table) == 2


def test_validation_corpus_must_equal_the_test_corpus(monkeypatch):
    monkeypatch.setitem(DETECTORS, Recording.name, Recording)
    with pytest.raises(ValueError, match="validation split's corpus differs"):
        statistic_all_test_questions_vs_validation_reference_by_severity(
            {**RUN_CONFIG, "failures": {"F1_shifted": RUN_CONFIG["failures"]["F1_shifted"]},
             "detectors": [Recording.name]}, {}, loader=with_validation(30),
            make_retriever=lambda: Retriever("clean/model", encoder=HashEncoder()))


def test_test_and_validation_questions_must_not_overlap(monkeypatch):
    monkeypatch.setitem(DETECTORS, Recording.name, Recording)
    with pytest.raises(ValueError, match="1 question ids are in both the test and validation splits"):
        statistic_all_test_questions_vs_validation_reference_by_severity(
            {**RUN_CONFIG, "failures": {"F1_shifted": RUN_CONFIG["failures"]["F1_shifted"]},
             "detectors": [Recording.name]}, {}, loader=with_validation(shared_questions=["q7"]),
            make_retriever=lambda: Retriever("clean/model", encoder=HashEncoder()))


def test_held_out_datasets_need_final():
    with pytest.raises(SystemExit, match="held out"):
        per_question_ndcg10_by_failure_severity_seed({"datasets": ["fiqa"], "failures": {}}, {})
