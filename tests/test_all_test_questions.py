import numpy as np
import pandas as pd
import pytest

from silent_rag import run as runner
from silent_rag.eval.all_test_questions import (
    mean_ndcg10_drop_all_test_questions_by_severity,
    per_question_ndcg10_by_failure_severity_seed,
)
from silent_rag.eval.ground_truth import bootstrap_ci
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.inject.f1_partial_upgrade import PartialUpgrade
from silent_rag.pipeline import Retriever
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


def test_mean_drop_is_a_hand_calculation_with_counts():
    got = mean_ndcg10_drop_all_test_questions_by_severity(per_question({"a": [0.1, 0.3], "b": [0.0, 0.2]}), BOOT).iloc[0]
    assert got["mean_ndcg10_drop"] == pytest.approx((0.2 + 0.1) / 2)  # question means 0.2 and 0.1
    assert (got["n_questions"], got["n_seeds"], got["injector"]) == (2, 2, "F1_partial_upgrade")


def test_seeds_are_averaged_before_the_question_bootstrap():
    # questions differ (0.0 to 0.3); seeds of a question barely do. Five seeds per question.
    drops = {f"q{i}": [0.1 * i + 0.001 * s for s in range(5)] for i in range(4)}
    got = mean_ndcg10_drop_all_test_questions_by_severity(per_question(drops), BOOT).iloc[0]
    question_means = np.array([np.mean(d) for d in drops.values()])
    assert (got["ci_low"], got["ci_high"]) == pytest.approx(bootstrap_ci(question_means, n_boot=2000, seed=0)[1:])
    pooled = bootstrap_ci(np.concatenate(list(drops.values())), n_boot=2000, seed=0)  # 20 "independent" values
    assert got["ci_high"] - got["ci_low"] > pooled[2] - pooled[1]


def test_unequal_seed_counts_are_an_error():
    with pytest.raises(ValueError, match="different numbers of seeds"):
        mean_ndcg10_drop_all_test_questions_by_severity(per_question({"a": [0.1, 0.3], "b": [0.0]}), BOOT)


def test_held_out_datasets_need_final():
    with pytest.raises(SystemExit, match="held out"):
        per_question_ndcg10_by_failure_severity_seed({"datasets": ["fiqa"], "failures": {}}, {})
