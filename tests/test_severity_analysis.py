"""Offline tests for Issue #16. No model downloads, no external data."""

import numpy as np
import pandas as pd
import pytest

from silent_rag.eval.severity_analysis import (
    adjacent_severity_trends, control_validation, interval_note, monotonicity_report,
    plot_severity_curves, prepare_windows, severity_summary,
)


def toy_results(n_seeds=3, detectors=("ks_top1", "re_embed")):
    rows = []
    for seed in range(n_seeds):
        for trial in range(4):
            for failure, severities, harmful in (
                ("clean", [0.0], False),
                ("F1_partial_upgrade", [0.1, 0.5, 0.9], True),
                ("H1_harmless", [0.1, 0.5, 0.9], False),
            ):
                for sev in severities:
                    drop = (sev * 0.1 + seed * 0.001 + trial * 0.0001) if harmful else 0.0
                    for detector in detectors:
                        rows.append({
                            "run_id": "same-run", "dataset": "nfcorpus", "model": "model",
                            "model_revision": None, "window_size": 100,
                            "failure": failure, "severity": sev, "seed": seed,
                            "trial": trial, "detector": detector,
                            "expected_harmful": harmful,
                            "ndcg_drop": drop,
                        })
    return pd.DataFrame(rows)


def test_two_detector_rows_count_as_only_one_window():
    raw = toy_results()
    windows = prepare_windows(raw)
    assert len(windows) * 2 == len(raw)
    summary = severity_summary(windows, n_boot=100, seed=2)
    assert len(summary) == 7
    assert set(summary.n_windows) == {12}
    assert set(summary.n_seeds) == {3}
    assert summary[summary.failure == "F1_partial_upgrade"].mean_drop.is_monotonic_increasing


def test_single_seed_has_no_pseudoreplicated_ci():
    windows = prepare_windows(toy_results(n_seeds=1))
    summary = severity_summary(windows, n_boot=100)
    assert summary.ci_low.isna().all() and summary.ci_high.isna().all()
    assert set(summary.n_seeds) == {1}


def test_mixed_detector_labels_or_drops_are_rejected():
    raw = toy_results()
    raw.loc[1, "ndcg_drop"] = 0.15
    with pytest.raises(ValueError, match="disagree"):
        prepare_windows(raw)


def test_duplicate_detector_window_rejected():
    raw = toy_results()
    with pytest.raises(ValueError, match="Duplicate rows"):
        prepare_windows(pd.concat([raw, raw.iloc[[0]]], ignore_index=True))


def test_heldout_dataset_rejected():
    raw = toy_results()
    raw.loc[0, "dataset"] = "scifact"
    with pytest.raises(ValueError, match="NFCorpus-only"):
        prepare_windows(raw)


def test_mixed_runs_rejected():
    raw = toy_results()
    raw.loc[0, "run_id"] = "different-run"
    with pytest.raises(ValueError, match="Mixed run_id"):
        prepare_windows(raw)


def make_windows(conditions, n_seeds=5, n_trials=4):
    """Window-level rows built directly. conditions: {(failure, severity, harmful): f(seed, trial) -> drop}."""
    rows = []
    for (failure, severity, harmful), drop in conditions.items():
        for seed in range(n_seeds):
            for trial in range(n_trials):
                rows.append({
                    "run_id": "same-run", "dataset": "nfcorpus", "model": "model",
                    "model_revision": None, "window_size": 100,
                    "failure": failure, "severity": severity, "seed": seed, "trial": trial,
                    "expected_harmful": harmful, "ndcg_drop": float(drop(seed, trial)),
                })
    return pd.DataFrame(rows)


def wobble(seed, scale):
    """Deterministic seed-to-seed noise with mean 0 over seeds 0..4."""
    return scale * (seed - 2)


def test_decrease_is_classified_from_the_interval():
    raw = toy_results(detectors=("ks_top1",))
    raw.loc[(raw.failure == "F1_partial_upgrade") & (raw.severity == 0.9), "ndcg_drop"] = 0.01
    trends = adjacent_severity_trends(prepare_windows(raw), n_boot=100)
    f1 = trends[trends.failure == "F1_partial_upgrade"]
    assert list(f1.trend) == ["increasing", "decreasing"]
    assert "diagnostic" not in trends.columns  # the sign-of-the-mean label is gone
    report = monotonicity_report(trends)
    assert list(report.failure) == ["F1_partial_upgrade"]  # controls get their verdict elsewhere
    assert report.range_verdict.iloc[0] == "non_monotonic"


def test_a_mean_that_moves_inside_the_noise_is_inconclusive():
    # the mean rises by 0.001, but seeds disagree on the sign: -0.019 ... +0.021
    windows = make_windows({
        ("F1", 0.1, True): lambda s, t: 0.05,
        ("F1", 0.5, True): lambda s, t: 0.051 + wobble(s, 0.01),
    })
    row = adjacent_severity_trends(windows, n_boot=200).iloc[0]
    assert row.mean_increase > 0
    assert row.ci_low < 0 < row.ci_high
    assert row.trend == "inconclusive"
    assert monotonicity_report(adjacent_severity_trends(windows, n_boot=200)).range_verdict.iloc[0] == "inconclusive"


def test_full_replacement_is_reported_apart_from_the_range():
    # damage grows from 0.1 to 0.9 and falls at 1.0, like a finished swap to a similar model
    drops = {0.1: 0.01, 0.5: 0.04, 0.9: 0.06, 1.0: 0.03}
    windows = make_windows({
        ("F1", sev, True): (lambda s, t, d=d: d + wobble(s, 0.001)) for sev, d in drops.items()
    })
    trends = adjacent_severity_trends(windows, n_boot=200)
    assert list(trends.scope) == ["range", "range", "endpoint"]
    assert list(trends.trend) == ["increasing", "increasing", "decreasing"]
    report = monotonicity_report(trends).iloc[0]
    assert report.range_verdict == "increasing"  # the 1.0 dip does not make the range non-monotonic
    assert report.n_range_steps == 2
    assert report.endpoint_trend == "decreasing"
    assert report.endpoint_mean_increase == pytest.approx(-0.03)
    # opting out folds it back in
    folded = adjacent_severity_trends(windows, n_boot=200, endpoint_severity=None)
    assert set(folded.scope) == {"range"}
    assert monotonicity_report(folded).range_verdict.iloc[0] == "non_monotonic"


def test_t_interval_is_the_main_interval_and_bootstrap_is_kept():
    from scipy import stats

    windows = make_windows({("F1", 0.5, True): lambda s, t: 0.05 + wobble(s, 0.01)})
    row = severity_summary(windows, n_boot=500).iloc[0]
    seed_means = np.array([0.05 + wobble(s, 0.01) for s in range(5)])
    half = stats.t.ppf(0.975, 4) * seed_means.std(ddof=1) / np.sqrt(5)
    assert row.ci_low == pytest.approx(0.05 - half)
    assert row.ci_high == pytest.approx(0.05 + half)
    # the comparison interval is there, and is a different calculation
    assert np.isfinite(row.boot_ci_low) and np.isfinite(row.boot_ci_high)
    assert row.boot_ci_low < row.mean_drop < row.boot_ci_high
    assert (row.boot_ci_low, row.boot_ci_high) != (row.ci_low, row.ci_high)


def test_intervals_use_seed_means_not_windows():
    # 4 trials or 40 per seed: same seed means, so the same interval
    few = make_windows({("F1", 0.5, True): lambda s, t: 0.05 + wobble(s, 0.01)}, n_trials=4)
    many = make_windows({("F1", 0.5, True): lambda s, t: 0.05 + wobble(s, 0.01)}, n_trials=40)
    a, b = severity_summary(few, n_boot=100).iloc[0], severity_summary(many, n_boot=100).iloc[0]
    assert (a.ci_low, a.ci_high) == pytest.approx((b.ci_low, b.ci_high))
    assert (a.n_windows, b.n_windows) == (20, 200)


def test_one_seed_gives_a_note_and_no_verdicts():
    windows = prepare_windows(toy_results(n_seeds=1))
    summary = severity_summary(windows, n_boot=100)
    note = interval_note(summary)
    assert "one seed" in note and "seeds:" in note
    assert summary.boot_ci_low.isna().all()
    trends = adjacent_severity_trends(windows, n_boot=100)
    assert set(trends.trend) == {"no_interval"} and set(trends.boot_trend) == {"no_interval"}
    assert set(monotonicity_report(trends).range_verdict) == {"no_interval"}
    assert set(control_validation(windows, margin=0.02, by_construction=["H1_harmless"]).verdict) == {"not_tested"}
    assert interval_note(severity_summary(prepare_windows(toy_results()), n_boot=100)) is None


def control_windows():
    return make_windows({
        ("clean", 0.0, False): lambda s, t: 0.0,
        ("F1", 0.5, True): lambda s, t: 0.08,
        ("H_flat", 0.1, False): lambda s, t: 0.001 + wobble(s, 0.001),
        ("H_flat", 0.9, False): lambda s, t: -0.002 + wobble(s, 0.001),
        ("H_better", 1.0, False): lambda s, t: -0.05 + wobble(s, 0.002),   # one severity, improves
        ("H_bad", 0.5, False): lambda s, t: 0.06 + wobble(s, 0.002),
        ("H_noisy", 0.5, False): lambda s, t: 0.015 + wobble(s, 0.01),
        ("H_shift", 0.5, False): lambda s, t: 0.0,                          # zero by construction
        ("H_silent_zero", 0.5, False): lambda s, t: 0.0,                    # zero, and nobody said why
    })


def test_control_validation_verdicts():
    out = control_validation(control_windows(), margin=0.02, by_construction=["H_shift"])
    verdict = {(r.failure, r.severity): (r.evidence, r.verdict) for r in out.itertuples()}
    assert verdict[("H_flat", 0.1)] == ("measured", "equivalent")
    assert verdict[("H_flat", 0.9)] == ("measured", "equivalent")
    assert verdict[("H_better", 1.0)] == ("measured", "not_worse")  # harmless, but not flat
    assert verdict[("H_bad", 0.5)] == ("measured", "exceeds_margin")
    assert verdict[("H_noisy", 0.5)] == ("measured", "inconclusive")
    assert set(out.failure) == {"H_flat", "H_better", "H_bad", "H_noisy", "H_shift", "H_silent_zero"}
    assert set(out.margin) == {0.02}


def test_zero_by_construction_is_not_counted_as_evidence():
    out = control_validation(control_windows(), margin=0.02, by_construction=["H_shift"])
    by = out.set_index("failure")
    assert (by.loc["H_shift", "evidence"], by.loc["H_shift", "verdict"]) == ("by_construction", "not_tested")
    # an exact zero nobody declared is flagged, not passed
    assert (by.loc["H_silent_zero", "evidence"], by.loc["H_silent_zero", "verdict"]) == ("all_zero_unexplained", "not_tested")
    with pytest.raises(ValueError, match="declared zero by construction"):
        control_validation(control_windows(), margin=0.02, by_construction=["H_flat"])
    with pytest.raises(ValueError, match="not controls in this run"):
        control_validation(control_windows(), margin=0.02, by_construction=["F1"])
    with pytest.raises(ValueError, match="collection of failure labels"):
        control_validation(control_windows(), margin=0.02, by_construction="H_shift")


def test_single_severity_control_is_validated_without_a_trend():
    windows = control_windows()
    out = control_validation(windows, margin=0.02, by_construction=["H_shift"])
    better = out[out.failure == "H_better"].iloc[0]
    assert better.n_severities == 1 and better.verdict == "not_worse"
    assert set(out[out.failure == "H_flat"].n_severities) == {2}
    trends = adjacent_severity_trends(windows, n_boot=100)
    assert "H_better" not in set(trends.failure)  # nothing to compare it with
    assert "H_flat" in set(trends.failure)


def test_margin_is_explicit_and_changes_the_verdict():
    windows = control_windows()
    with pytest.raises(TypeError):
        control_validation(windows)  # no default margin
    for bad in (0, -0.02, float("nan"), float("inf"), True, "0.02"):
        with pytest.raises(ValueError, match="margin"):
            control_validation(windows, margin=bad)

    def verdict(margin):
        out = control_validation(windows, margin=margin, by_construction=["H_shift"])
        return out.set_index("failure").loc["H_bad", "verdict"]

    assert verdict(0.02) == "exceeds_margin"
    assert verdict(0.1) == "equivalent"  # same data, a looser margin: the margin is the claim


def test_plot_keeps_the_endpoint_in_the_figure(tmp_path):
    drops = {0.1: 0.01, 0.5: 0.04, 0.9: 0.06, 1.0: 0.03}
    windows = make_windows({
        **{("F1", sev, True): (lambda s, t, d=d: d + wobble(s, 0.001)) for sev, d in drops.items()},
        ("H_better", 1.0, False): lambda s, t: -0.05 + wobble(s, 0.002),
    })
    paths = plot_severity_curves(severity_summary(windows, n_boot=100), tmp_path)
    assert {p.name for p in paths} == {"severity_F1.png", "severity_H_better.png"}
    assert all(p.stat().st_size > 1000 for p in paths)


def test_adjacent_comparisons_need_paired_windows():
    windows = prepare_windows(toy_results())
    windows = windows.loc[~((windows.failure == "F1_partial_upgrade") &
                            (windows.severity == 0.9) & (windows.seed == 0) &
                            (windows.trial == 0))]
    with pytest.raises(ValueError, match="lack paired"):
        adjacent_severity_trends(windows, n_boot=100)


def test_plot_writes_a_figure_per_injection(tmp_path):
    windows = prepare_windows(toy_results())
    summary = severity_summary(windows, n_boot=100)
    paths = plot_severity_curves(summary, tmp_path)
    assert len(paths) == 2
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)
    assert {p.name for p in paths} == {"severity_F1_partial_upgrade.png", "severity_H1_harmless.png"}


def test_clean_is_not_a_harmful_condition():
    raw = toy_results()
    raw.loc[raw.failure == "clean", "expected_harmful"] = True
    with pytest.raises(ValueError, match="Clean must"):
        prepare_windows(raw)


def test_bootstrap_is_deterministic():
    windows = prepare_windows(toy_results())
    a = severity_summary(windows, n_boot=200, seed=17)
    b = severity_summary(windows, n_boot=200, seed=17)
    pd.testing.assert_frame_equal(a, b)
