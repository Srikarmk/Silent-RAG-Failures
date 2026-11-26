import numpy as np
import pytest

from silent_rag.eval.ground_truth import ndcg_at_k, paired_bootstrap, window_has_significant_ndcg10_drop
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


def test_significance_rule_boundaries():
    # p equal to alpha is not significant; a drop equal to the minimum counts
    assert window_has_significant_ndcg10_drop(0.05, 0.1, alpha=0.05, min_mean_ndcg10_drop=0.0) is False
    assert window_has_significant_ndcg10_drop(0.01, 0.02, alpha=0.05, min_mean_ndcg10_drop=0.02) is True
    assert window_has_significant_ndcg10_drop(0.01, 0.019, alpha=0.05, min_mean_ndcg10_drop=0.02) is False


def test_significance_rule_matches_the_runners_inline_rule():
    # the rule run.py applies today: p < alpha and drop >= min_drop
    values = [(p, d) for p in (0.0, 0.0499, 0.05, 0.5, float("nan")) for d in (-0.1, 0.0, 0.02, 0.3, float("nan"))]
    for p, d in values:
        got = window_has_significant_ndcg10_drop(p, d, alpha=0.05, min_mean_ndcg10_drop=0.02)
        assert type(got) is bool
        assert got == (p < 0.05 and d >= 0.02), (p, d)


def test_significance_rule_on_arrays():
    p = np.array([0.01, 0.05, 0.01, 0.2])
    drop = np.array([0.1, 0.1, 0.0, 0.1])
    got = window_has_significant_ndcg10_drop(p, drop, alpha=0.05, min_mean_ndcg10_drop=0.02)
    assert got.dtype == bool
    np.testing.assert_array_equal(got, [True, False, False, False])


def test_held_out_needs_final_flag():
    check_datasets(["nfcorpus"], final=False)
    with pytest.raises(SystemExit):
        check_datasets(["nfcorpus", "scifact"], final=False)
    check_datasets(["scifact"], final=True)
