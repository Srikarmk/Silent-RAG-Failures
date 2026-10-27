"""Synthetic, deterministic regression tests for Issue #17 detector metrics."""

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from silent_rag.eval.detector_scoring import (
    bootstrap_auroc_ci,
    compute_auroc,
    earliest_detectable_severity,
    false_alarm_rate,
    negative_control_report,
    severity_detection_table,
)
from silent_rag.eval.results import ResultRow, to_frame


BASE = {
    "run_id": "run0",
    "dataset": "nfcorpus",
    "model": "baseline",
    "model_revision": "pinned",
    "seed": 0,
    "window_size": 100,
    "detector": "ks_top1",
    "failure": "clean",
    "severity": 0.0,
    "expected_harmful": False,
    "harmful": False,
    "score": 0.0,
}


def make_df(*items):
    return pd.DataFrame([{**BASE, "trial": i, **item} for i, item in enumerate(items)])


def obs(failure, score, harmful, severity=0.5, **extra):
    return {"failure": failure, "score": score, "harmful": harmful,
            "severity": severity,
            "expected_harmful": failure not in ("clean", "H1", "H2", "H3"),
            **extra}


def test_auroc_drops_nonharmful_target_windows():
    # the F1 window whose drop wasn't significant is neither positive nor negative
    df = make_df(obs("clean", .1, False), obs("H1", .9, False),
                 obs("F1", .95, False), obs("F1", .8, True))
    assert compute_auroc(df, "ks_top1", "F1") == pytest.approx(1 / 2)
    perfect = make_df(obs("clean", .1, False), obs("F1", .05, False), obs("F1", .8, True))
    assert compute_auroc(perfect, "ks_top1", "F1") == 1.0


def test_auroc_counts_clean_windows_as_negatives_even_if_labelled_harmful():
    # clean windows are undamaged; a significant drop there is a ground-truth false positive
    df = make_df(obs("clean", .1, True), obs("clean", .2, False), obs("F1", .8, True))
    assert compute_auroc(df, "ks_top1", "F1") == 1.0


def test_auroc_excludes_failed_controls_from_f1_positives():
    df = make_df(obs("clean", .1, False), obs("H1", .05, True),
                 obs("F1", .2, False), obs("F1", .8, True))
    assert compute_auroc(df, "ks_top1", "F1") == pytest.approx(1.0)


def test_auroc_requires_both_classes():
    df = make_df(obs("clean", .1, False), obs("F1", .2, False))
    with pytest.raises(ValueError, match="both harmful and non-harmful"):
        compute_auroc(df, "ks_top1", "F1")


@pytest.mark.parametrize("scores,expected", [
    ([.1, .2, .8, .9], 1.0),
    ([.5, .5, .5, .5], .5),
    ([.8, .9, .1, .2], 0.0),
])
def test_auroc_ties_and_direction(scores, expected):
    df = make_df(obs("clean", scores[0], False), obs("clean", scores[1], False),
                 obs("F1", scores[2], True), obs("F1", scores[3], True))
    assert compute_auroc(df, "ks_top1", "F1") == pytest.approx(expected)


def test_auroc_rejects_nonfinite_scores():
    df = make_df(obs("clean", .1, False), obs("F1", np.nan, True))
    with pytest.raises(ValueError, match="finite"):
        compute_auroc(df, "ks_top1", "F1")


def test_mixed_datasets_and_models_are_rejected():
    df = make_df(obs("clean", .1, False), obs("F1", .8, True))
    for col, value in (("dataset", "fiqa"), ("model", "different"),
                       ("window_size", 150), ("model_revision", "new")):
        modified = df.copy()
        modified.loc[1, col] = value
        with pytest.raises(ValueError, match=f"Mixed {col}"):
            compute_auroc(modified, "ks_top1", "F1")


def test_mixed_runs_require_explicit_approval():
    df = make_df(obs("clean", .1, False), obs("F1", .9, True, run_id="run1", seed=1))
    with pytest.raises(ValueError, match="Mixed run_id"):
        compute_auroc(df, "ks_top1", "F1")
    assert compute_auroc(df, "ks_top1", "F1", allow_multiple_runs=True) == 1.0


def test_missing_detector_and_failure_are_errors():
    df = make_df(obs("clean", .1, False), obs("F1", .8, True))
    with pytest.raises(ValueError, match="No results for detector"):
        compute_auroc(df, "missing", "F1")
    with pytest.raises(ValueError, match="No rows for failure"):
        compute_auroc(df, "ks_top1", "F2")


def test_duplicate_rows_rejected():
    df = make_df(obs("clean", .1, False), obs("F1", .8, True))
    duplicate = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate ResultRows"):
        compute_auroc(duplicate, "ks_top1", "F1")


def seed_df(pairs):
    items = []
    for seed, (clean, harmed) in enumerate(pairs):
        items += [obs("clean", clean, False, seed=seed, run_id=f"run{seed}", trial=0),
                  obs("F1", harmed, True, seed=seed, run_id=f"run{seed}", trial=0)]
    return make_df(*items)


def test_bootstrap_ci_perfect_separation():
    df = seed_df([(.1, .8), (.2, .9), (.1, .9), (.2, .8)])
    auc, low, high = bootstrap_auroc_ci(
        df, "ks_top1", "F1", n_boot=200, seed=42, allow_multiple_runs=True)
    assert (auc, low, high) == pytest.approx((1.0, 1.0, 1.0))


def test_bootstrap_ci_reproducible():
    df = seed_df([(.1, .8), (.7, .2), (.4, .6), (.9, .5)])
    kwargs = dict(n_boot=300, seed=42, allow_multiple_runs=True)
    a = bootstrap_auroc_ci(df, "ks_top1", "F1", **kwargs)
    b = bootstrap_auroc_ci(df, "ks_top1", "F1", **kwargs)
    assert a == pytest.approx(b)
    assert a[0] == pytest.approx(.5)
    assert 0 <= a[1] <= a[2] <= 1


def test_bootstrap_ci_single_seed_unavailable():
    df = make_df(obs("clean", .1, False), obs("F1", .8, True))
    with pytest.raises(ValueError, match="at least two experiment blocks"):
        bootstrap_auroc_ci(df, "ks_top1", "F1")


def test_bootstrap_ci_does_not_silently_pool_runs():
    df = seed_df([(.1, .8), (.2, .9)])
    with pytest.raises(ValueError, match="Mixed run_id"):
        bootstrap_auroc_ci(df, "ks_top1", "F1")


@pytest.mark.parametrize("kw,err", [
    ({"n_boot": 0}, "n_boot"),
    ({"n_boot": 2.5}, "n_boot"),
    ({"alpha": 1.0}, "alpha"),
])
def test_bootstrap_ci_invalid_settings(kw, err):
    df = seed_df([(.1, .8), (.2, .9)])
    with pytest.raises(ValueError, match=err):
        bootstrap_auroc_ci(df, "ks_top1", "F1", allow_multiple_runs=True, **kw)


def test_negative_controls_report_both_metrics_and_harm_rate():
    df = make_df(obs("clean", .99, False), obs("H1", .8, False),
                 obs("H1", .1, False), obs("H2", .2, False),
                 obs("H3", .95, True), obs("F1", .6, True))
    report = negative_control_report(df, "ks_top1", threshold=.5)
    assert report["n_controls"] == 4
    assert report["n_harmful_controls"] == 1
    assert report["n_harmless_controls"] == 3
    assert report["control_alarm_rate"] == pytest.approx(.5)
    assert report["false_alarm_rate"] == pytest.approx(1 / 3)
    assert report["harmful_control_rate"] == pytest.approx(.25)
    assert false_alarm_rate(df, "ks_top1", threshold=.5) == pytest.approx(1 / 3)


def test_control_metrics_per_failure():
    df = make_df(obs("H1", .9, False), obs("H1", .1, False),
                 obs("H2", .9, False))
    assert false_alarm_rate(df, "ks_top1", .5, failure="H1") == .5
    assert false_alarm_rate(df, "ks_top1", .5, failure="H2") == 1.0


def test_false_alarm_boundary_equal_is_alarm():
    df = make_df(obs("clean", .9, False), obs("H1", .5, False),
                 obs("H1", .49, False), obs("H2", .2, False))
    assert false_alarm_rate(df, "ks_top1", .5) == pytest.approx(1 / 3)


def test_no_negative_controls():
    df = make_df(obs("clean", .1, False), obs("F1", .9, True))
    with pytest.raises(ValueError, match="No negative controls"):
        false_alarm_rate(df, "ks_top1", .5)


def test_all_controls_harmful_reports_unavailable_fpr():
    df = make_df(obs("H1", .9, True), obs("H1", .1, True))
    assert negative_control_report(df, "ks_top1", .5)["false_alarm_rate"] is None
    with pytest.raises(ValueError, match="No valid negative controls"):
        false_alarm_rate(df, "ks_top1", .5)


def test_bad_thresholds():
    df = make_df(obs("H1", .1, False), obs("F1", .8, True))
    with pytest.raises(ValueError, match="Threshold must be finite"):
        false_alarm_rate(df, "ks_top1", np.nan)
    with pytest.raises(ValueError, match="Threshold must be finite"):
        severity_detection_table(df, "ks_top1", "F1", np.inf)


def test_earliness_does_not_count_nonharmful_alarms():
    df = make_df(*(
        [obs("F1", s, h, severity=.1) for s, h in
         zip([.9, .9, .9, .9, .1], [False, False, False, False, True])]
        + [obs("F1", s, True, severity=.5) for s in [.6, .7, .8, .9, .2]]
    ))
    assert earliest_detectable_severity(df, "ks_top1", "F1", .5) == pytest.approx(.5)
    summary = severity_detection_table(df, "ks_top1", "F1", .5)
    assert summary.loc[summary.severity.eq(.1), "alarm_rate"].iloc[0] == .8
    assert summary.loc[summary.severity.eq(.1), "true_detection_rate"].iloc[0] == 0
    assert summary.loc[summary.severity.eq(.5), "true_detection_rate"].iloc[0] == .8


def test_earliness_is_not_capped_by_harmful_fraction():
    # only a quarter of windows are harmful, but the detector catches all of them
    df = make_df(*([obs("F1", .9, True, severity=.1)] * 5
                   + [obs("F1", .9, False, severity=.1)] * 15))
    summary = severity_detection_table(df, "ks_top1", "F1", .5)
    assert summary.iloc[0]["tpr_when_harmful"] == 1.0
    assert summary.iloc[0]["harmful_fraction"] == .25
    assert summary.iloc[0]["true_detection_rate"] == .25
    assert earliest_detectable_severity(df, "ks_top1", "F1", .5) == pytest.approx(.1)


def test_earliness_must_hold_at_every_higher_severity():
    df = make_df(*([obs("F1", .9, True, severity=.1)] * 5
                   + [obs("F1", .1, True, severity=.3)] * 5
                   + [obs("F1", .9, True, severity=.5)] * 5))
    assert earliest_detectable_severity(df, "ks_top1", "F1", .5) == pytest.approx(.5)


def test_earliness_skips_severities_without_enough_harmful_windows():
    df = make_df(*([obs("F1", .9, True, severity=.1)] * 5
                   + [obs("F1", .1, True, severity=.3)] * 2
                   + [obs("F1", .1, False, severity=.3)] * 3
                   + [obs("F1", .9, True, severity=.5)] * 5))
    assert earliest_detectable_severity(df, "ks_top1", "F1", .5) == pytest.approx(.1)


def test_earliness_handles_unordered_severities():
    df = make_df(*([obs("F1", .9, True, severity=.9)] * 5
                   + [obs("F1", .1, True, severity=.1)] * 5
                   + [obs("F1", .9, True, severity=.5)] * 4
                   + [obs("F1", .1, True, severity=.5)]))
    assert earliest_detectable_severity(df, "ks_top1", "F1", .5) == .5


def test_earliness_returns_none_for_low_power_or_insufficient_harm():
    df = make_df(*[obs("F1", s, True, severity=.8) for s in [.1, .2, .3, .9, .9]])
    assert earliest_detectable_severity(df, "ks_top1", "F1", .5) is None
    few = make_df(*([obs("F1", .9, True)] * 4 + [obs("F1", .9, False)] * 2))
    assert earliest_detectable_severity(few, "ks_top1", "F1", .5) is None


def test_earliness_correct_detector_and_failure():
    df = make_df(*([obs("F1", .1, True)] * 5
                   + [obs("F1", .9, True, detector="other")] * 5
                   + [obs("F2", .9, True)] * 5))
    assert earliest_detectable_severity(df, "ks_top1", "F1", .5) is None


@pytest.mark.parametrize("kw", [
    {"min_detection_rate": 0}, {"min_detection_rate": 1.1},
    {"min_harmful_windows": 0}, {"min_harmful_windows": True},
])
def test_earliness_invalid_settings(kw):
    df = make_df(*[obs("F1", .9, True)] * 5)
    with pytest.raises(ValueError):
        earliest_detectable_severity(df, "ks_top1", "F1", .5, **kw)


def test_earliness_threshold_boundary():
    df = make_df(*[obs("F1", score, True, severity=.3) for score in [.5] * 4 + [.1]])
    assert earliest_detectable_severity(df, "ks_top1", "F1", .5) == pytest.approx(.3)


def test_real_resultrow_schema_integration():
    available = {f.name for f in fields(ResultRow)}
    def record(condition, score, harmful, trial):
        kwargs = dict(run_id="run0", dataset="nfcorpus", model="baseline",
                      model_revision="pinned", failure=condition,
                      expected_harmful=(condition == "F1"),
                      severity=0.5 if condition != "clean" else 0.,
                      seed=0, trial=trial, window_size=100, detector="ks_top1",
                      score=score, ndcg_drop=.2 if harmful else 0.0,
                      harmful=harmful, cost_s=.01)
        if "p_value" in available:
            kwargs["p_value"] = 0.01 if harmful else 1.0
        return ResultRow(**kwargs)

    df = to_frame([
        record("clean", .1, False, 0), record("H1", .2, False, 0),
        record("F1", .9, True, 0), record("F1", .2, False, 1),
    ])
    assert compute_auroc(df, "ks_top1", "F1") == pytest.approx(1.0)
    assert false_alarm_rate(df, "ks_top1", .5) == 0.0
    assert severity_detection_table(df, "ks_top1", "F1", .5).iloc[0]["n_windows"] == 2
