"""Offline tests for Issue #16. No model downloads, no external data."""

import numpy as np
import pandas as pd
import pytest

from silent_rag.eval.severity_analysis import (
    adjacent_severity_trends, plot_severity_curves, prepare_windows, severity_summary,
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


def test_nonmonotonic_failure_is_flagged():
    raw = toy_results(detectors=("ks_top1",))
    raw.loc[(raw.failure == "F1_partial_upgrade") & (raw.severity == 0.9), "ndcg_drop"] = 0.01
    windows = prepare_windows(raw)
    trends = adjacent_severity_trends(windows, n_boot=100)
    f1 = trends[trends.failure == "F1_partial_upgrade"]
    assert list(f1.diagnostic) == ["mean_not_decreased", "mean_decreased"]
    controls = trends[trends.failure == "H1_harmless"]
    assert set(controls.diagnostic) == {"control_flatness_not_yet_established"}


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
