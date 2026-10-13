import numpy as np
import pytest

from silent_rag.eval.window import detection_rate, min_detectable_drop, simulate_damage


def recall_like(n=300, hit_rate=0.55, seed=0):
    """Per-query nDCG that looks like SciFact: mostly 0 or close to 1."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < hit_rate, rng.uniform(0.6, 1.0, n), 0.0)


def test_knockout_hits_the_requested_drop_on_average():
    clean = recall_like(5000)
    rng = np.random.default_rng(1)
    drops = [clean.mean() - simulate_damage(clean, 0.1, rng).mean() for _ in range(50)]
    assert np.mean(drops) == pytest.approx(0.1, abs=0.005)


def test_churn_keeps_the_mean_but_moves_queries():
    clean = recall_like(5000)
    damaged = simulate_damage(clean, 0.0, np.random.default_rng(2), churn=0.5)
    assert damaged.mean() == pytest.approx(clean.mean(), abs=0.02)
    assert (damaged > clean).any() and (damaged < clean).any()


def test_drop_bigger_than_the_mean_zeroes_the_window():
    damaged = simulate_damage(np.full(10, 0.1), 0.5, np.random.default_rng(0))
    np.testing.assert_array_equal(damaged, 0.0)
    with pytest.raises(ValueError):
        simulate_damage(np.full(10, 0.1), -0.1, np.random.default_rng(0))


def test_no_drop_means_no_alarm_without_churn():
    assert detection_rate(recall_like(), 50, 0.0, trials=30) == 0.0


def test_false_alarms_stay_near_alpha_with_churn():
    rate = detection_rate(recall_like(), 100, 0.0, trials=300, churn=0.5)
    assert rate < 0.12


def test_bigger_windows_catch_smaller_drops():
    pool = recall_like()
    small = detection_rate(pool, 25, 0.05, trials=100, churn=0.25)
    large = detection_rate(pool, 150, 0.05, trials=100, churn=0.25)
    assert large > small


def test_window_must_fit_in_the_pool():
    with pytest.raises(ValueError):
        detection_rate(recall_like(100), 101, 0.1)


def test_min_detectable_drop_picks_the_first_drop_over_power():
    rates = {0.0: 0.04, 0.02: 0.3, 0.04: 0.79, 0.06: 0.85, 0.08: 0.97}
    assert min_detectable_drop(rates, 0.8) == 0.06
    assert min_detectable_drop(rates, 0.5) == 0.04
    assert np.isnan(min_detectable_drop({0.0: 0.0, 0.1: 0.5}, 0.8))
