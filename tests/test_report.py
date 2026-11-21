import numpy as np
import pandas as pd
import pytest

from silent_rag.eval import report as R


def table(seeds=(0, 1, 2), trials=10):
    """A good detector `a` and a useless one `b`, one harmful failure and one control."""
    rng = np.random.default_rng(0)
    rows = []
    for seed in seeds:
        for t in range(trials):
            for failure, sev, harm, drop in [("clean", 0.0, False, 0.0), ("F1_x", 0.5, True, 0.05),
                                             ("H_y", 1.0, False, 0.0)]:
                # b's score depends only on the window, so it can't tell the conditions apart
                same = float(np.random.default_rng([seed, t]).normal())
                for det in ("a", "b"):
                    score = rng.normal(4.0 if failure == "F1_x" else 0.0) if det == "a" else same
                    rows.append(dict(run_id="r1", dataset="nfcorpus", model="m", model_revision="x" * 40,
                                     failure=failure, expected_harmful=harm, severity=sev, seed=seed, trial=t,
                                     window_size=100, detector=det, score=score, ndcg_drop=drop,
                                     p_value=0.01 if drop > 0 else 0.5, harmful=drop > 0,
                                     cost_s=0.0, cost_usd=0.0))
    return pd.DataFrame(rows)


def test_harmful_is_recomputed_for_each_min_drop():
    df = table()
    assert R.relabel(df, 0.02).harmful.sum() == len(df[df.failure == "F1_x"])
    assert R.relabel(df, 0.06).harmful.sum() == 0


def test_auroc_separates_a_good_detector_from_a_useless_one():
    a = R.auroc_table(R.relabel(table(trials=40), 0.02), n_boot=200).set_index("detector")
    assert a.loc["a", "auroc"] > 0.9 and a.loc["a", "lo"] <= a.loc["a", "auroc"] <= a.loc["a", "hi"]
    assert a.loc["b", "auroc"] == 0.5


def test_one_seed_gives_auroc_without_a_ci():
    a = R.auroc_table(R.relabel(table(seeds=(0,)), 0.02), n_boot=50)
    assert a.auroc.notna().all() and a.lo.isna().all()


def test_no_harmful_windows_leaves_a_gap_not_an_error():
    a = R.auroc_table(R.relabel(table(), 0.06), n_boot=50)
    assert a.auroc.isna().all()


def test_false_alarms_on_controls_near_the_clean_rate():
    t = R.alarm_table(R.relabel(table(trials=40), 0.02)).set_index("detector")
    assert t.loc["a", "false alarms: H_y"] < 0.2
    assert t.loc["a", "earliest: F1_x"] == 0.5


def test_report_has_every_section():
    text = R.report(table(), min_drops=(0.02,), n_boot=50)
    for part in ("# Results: run r1", "## Cost per check", "## min_drop 0.02", "How often", "AUROC", "False alarms"):
        assert part in text


def test_report_refuses_mixed_runs():
    df = table()
    df.loc[df.index[:5], "run_id"] = "r2"
    with pytest.raises(SystemExit, match="one run per report"):
        R.report(df)


def test_cost_table_counts_unknown_times():
    df = table()
    df.loc[df.detector.eq("b"), "cost_s"] = 0.002
    df.loc[df.index[df.detector.eq("b")][:3], "cost_s"] = np.nan
    t = R.cost_table(df).set_index("detector")
    assert t.loc["b", "median ms"] == pytest.approx(2.0) and t.loc["b", "unknown"] == 3
    assert t.loc["a", "unknown"] == 0
