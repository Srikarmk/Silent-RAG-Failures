import numpy as np
import pytest

from silent_rag.eval.ground_truth import ndcg_at_k, paired_bootstrap
from silent_rag.run import check_datasets


def test_ndcg_perfect_and_miss():
    assert ndcg_at_k(["a", "b"], {"a": 1}) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], {"a": 1}) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k(["x", "y"], {"a": 1}) == 0.0


def test_bootstrap_sees_a_real_drop():
    clean = np.ones(200)
    drop, p = paired_bootstrap(clean, clean - 0.2)
    assert drop == pytest.approx(0.2)
    assert p < 0.05


def test_held_out_needs_final_flag():
    check_datasets(["nfcorpus"], final=False)
    with pytest.raises(SystemExit):
        check_datasets(["nfcorpus", "scifact"], final=False)
    check_datasets(["scifact"], final=True)
