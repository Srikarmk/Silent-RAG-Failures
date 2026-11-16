import pandas as pd
import pytest

from silent_rag import run as runner
from silent_rag.eval.all_test_questions import per_question_ndcg10_by_failure_severity_seed
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


def test_held_out_datasets_need_final():
    with pytest.raises(SystemExit, match="held out"):
        per_question_ndcg10_by_failure_severity_seed({"datasets": ["fiqa"], "failures": {}}, {})
