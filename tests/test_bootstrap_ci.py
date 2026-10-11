import numpy as np
import pytest

from silent_rag.eval.ground_truth import bootstrap_ci


def test_constant_scores_give_a_zero_width_ci():
    mean, lo, hi = bootstrap_ci(np.full(50, 0.4))
    assert mean == lo == hi == pytest.approx(0.4)


def test_single_score():
    assert bootstrap_ci([0.7]) == pytest.approx((0.7, 0.7, 0.7))


def test_ci_brackets_the_mean_and_accepts_lists():
    scores = [0.0, 0.2, 0.5, 1.0, 1.0, 0.3, 0.0, 0.8]
    mean, lo, hi = bootstrap_ci(scores)
    assert mean == pytest.approx(np.mean(scores))
    assert lo <= mean <= hi


def test_same_seed_same_ci():
    scores = np.random.default_rng(1).random(100)
    assert bootstrap_ci(scores, seed=3) == bootstrap_ci(scores, seed=3)
    assert bootstrap_ci(scores, seed=3) != bootstrap_ci(scores, seed=4)


def test_width_matches_normal_approximation():
    # bernoulli(0.5), n=400: se = 0.025, so a 95% ci is about +/- 0.049
    scores = np.random.default_rng(0).integers(0, 2, 400).astype(float)
    mean, lo, hi = bootstrap_ci(scores)
    se = scores.std() / np.sqrt(len(scores))
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.1)


def test_smaller_alpha_gives_a_wider_ci():
    scores = np.random.default_rng(2).random(200)
    _, lo95, hi95 = bootstrap_ci(scores, alpha=0.05)
    _, lo80, hi80 = bootstrap_ci(scores, alpha=0.20)
    assert hi95 - lo95 > hi80 - lo80


@pytest.mark.parametrize(
    "scores, kwargs",
    [
        ([], {}),
        ([[0.1, 0.2], [0.3, 0.4]], {}),
        ([0.1, np.nan], {}),
        ([0.1, np.inf], {}),
        ([0.1, 0.2], {"alpha": 0}),
        ([0.1, 0.2], {"alpha": 1}),
        ([0.1, 0.2], {"n_boot": 0}),
        ([0.1, 0.2], {"n_boot": 2.5}),
    ],
)
def test_bad_input_raises(scores, kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci(scores, **kwargs)
