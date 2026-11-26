import numpy as np
import pandas as pd
import pytest

from silent_rag.eval import two_stage


def results(seeds=range(6)):
    """A cheap detector that fires on the failure and the control, and an expensive one
    that fires only on the failure, like a quality check would."""
    rng = np.random.default_rng(0)
    rows = []
    for seed in seeds:
        for failure, severity, harmful, shift in [("clean", 0.0, False, (0, 0)), ("F_x", 0.5, True, (4, 4)),
                                                  ("H_x", 1.0, False, (4, 0))]:
            for trial in range(40):
                for det, s, cost in [("cheap", shift[0], 0.001), ("check", shift[1], 2.0)]:
                    rows.append({"run_id": "r", "dataset": "nfcorpus", "seed": seed, "failure": failure,
                                 "severity": severity, "trial": trial, "detector": det,
                                 "score": s + rng.normal(), "cost_s": cost, "expected_harmful": harmful,
                                 "p_value": 0.001 if harmful else 0.5, "ndcg_drop": 0.05 if harmful else 0.0})
    return pd.DataFrame(rows)


def test_pair_keeps_detection_drops_the_control_false_alarm_and_costs_less():
    wide = two_stage.scores(results(), ["cheap", "check"])
    table = two_stage.evaluate(wide, "cheap", "check", calibration_seeds=[0, 1, 2])
    f, h = table.loc[("F_x", 0.5)], table.loc[("H_x", 1.0)]
    assert f.stage1 > 0.95 and f.pair > 0.95
    assert h.stage1 > 0.95 and h.pair < 0.1
    s = two_stage.summary(table)
    assert s.pair_false_alarm < s.stage1_false_alarm
    # stage 2 runs only after stage 1 fires: on 2 of 3 conditions here
    assert s.stage1_cost_s < s.pair_cost_s < s.stage2_cost_s


def test_cost_counts_stage_two_only_when_stage_one_fires():
    wide = two_stage.scores(results(), ["cheap", "check"])
    table = two_stage.evaluate(wide, "cheap", "check", calibration_seeds=[0, 1, 2])
    clean = table.loc[("clean", 0.0)]
    assert clean.cost_pair_s == pytest.approx(0.001 + clean.stage1 * 2.0)


def test_thresholds_come_from_calibration_clean_windows_only():
    wide = two_stage.scores(results(), ["cheap", "check"])
    shifted = wide.copy()
    shifted.loc[~shifted.failure.eq("clean"), "cheap"] += 100  # failures and controls don't move it
    assert two_stage.threshold(shifted, "cheap", [0, 1]) == two_stage.threshold(wide, "cheap", [0, 1])
    late = wide.copy()
    late.loc[late.seed >= 2, "cheap"] += 100  # test seeds don't move it either
    assert two_stage.threshold(late, "cheap", [0, 1]) == two_stage.threshold(wide, "cheap", [0, 1])


def test_harmful_failures_are_scored_on_harmful_windows_only():
    df = results()
    df.loc[(df.failure == "F_x") & (df.trial < 10), "p_value"] = 0.5  # damaged but not significant
    table = two_stage.evaluate(two_stage.scores(df, ["cheap", "check"]), "cheap", "check", [0, 1, 2])
    assert table.loc[("F_x", 0.5)].windows == 3 * 30


def test_evaluation_needs_test_seeds_and_known_detectors():
    wide = two_stage.scores(results(), ["cheap", "check"])
    with pytest.raises(ValueError, match="leave some seeds"):
        two_stage.evaluate(wide, "cheap", "check", calibration_seeds=range(6))
    with pytest.raises(ValueError, match=r"no rows for \['llm_judge'\]"):
        two_stage.scores(results(), ["cheap", "llm_judge"])


def test_stage1_matched_runs_at_the_pairs_clean_false_alarm_rate():
    wide = two_stage.scores(results(), ["cheap", "check"])
    cal = [0, 1, 2]
    t1 = two_stage.threshold(wide, "cheap", cal)
    t1m = two_stage.matched_threshold(wide, "cheap", "check", cal)
    assert t1m >= t1  # the AND gate is stricter, so stage 1 has to be too
    clean = wide[wide.failure.eq("clean") & wide.seed.isin(cal)]
    pair_rate = ((clean.cheap > t1) & (clean.check > two_stage.threshold(wide, "check", cal))).mean()
    assert (clean.cheap > t1m).mean() <= pair_rate + 1 / len(clean)
    table = two_stage.evaluate(wide, "cheap", "check", cal)
    # here stage 2 really adds information: on the control a stricter stage 1 still fires, the pair doesn't
    h = table.loc[("H_x", 1.0)]
    assert h.stage1_matched > h.pair


def test_control_windows_that_did_harm_are_not_false_alarms():
    df = results()
    hit = (df.failure == "H_x") & (df.trial < 10)
    df.loc[hit, "p_value"], df.loc[hit, "ndcg_drop"] = 0.001, 0.05
    table = two_stage.evaluate(two_stage.scores(df, ["cheap", "check"]), "cheap", "check", [0, 1, 2])
    assert table.loc[("H_x", 1.0)].windows == 3 * 30


def test_by_seed_splits_rows_per_test_seed():
    wide = two_stage.scores(results(), ["cheap", "check"])
    table = two_stage.evaluate(wide, "cheap", "check", [0, 1, 2], by_seed=True)
    assert sorted(table.index.get_level_values("seed").unique()) == [3, 4, 5]
    assert table.loc[(3, "F_x", 0.5)].windows == 40


def test_summary_pools_windows():
    table = pd.DataFrame({"expected_harmful": [False, False], "windows": [10, 30], "stage1": [0.0, 1.0],
                          "stage2": [0.0, 0.0], "pair": [0.0, 0.0], "stage1_matched": [0.0, 0.0],
                          "cost_stage1_s": [1.0, 1.0], "cost_stage2_s": [1.0, 1.0], "cost_pair_s": [1.0, 1.0]},
                         index=pd.MultiIndex.from_tuples([("clean", 0.0), ("H_x", 1.0)], names=["failure", "severity"]))
    assert two_stage.summary(table).stage1_false_alarm == 0.75


def test_scores_refuses_duplicate_windows():
    df = results()
    with pytest.raises(ValueError, match="same window twice"):
        two_stage.scores(pd.concat([df, df.head(1)]), ["cheap", "check"])

def gated_rows():
    """two_stage scores as the gated detector writes them: -1 when stage 1 didn't flag."""
    rows = []
    for failure, harm, drop, scores in [
        ("clean", False, 0.0, [-1, -1, 0.70, 0.72]),     # two judged, neither above 0.75
        ("F1_x", True, 0.05, [0.80, 0.90, -1, 0.74]),    # three judged, two alarm
        ("F1_x", True, 0.001, [0.95, -1, -1, -1]),       # tiny drop: not harmful at min_drop 0.02
        ("H_y", False, 0.0, [0.76, -1, -1, 0.0]),        # one false alarm; the last was judged and scored 0
        ("H_y", False, 0.05, [0.90]),                    # a control window that did harm: not a false alarm
    ]:
        for s in scores:
            rows.append(dict(run_id="r", dataset="nfcorpus", seed=7, failure=failure, severity=0.5 if harm else 0.0,
                             trial=len(rows), detector="two_stage", score=s,
                             cost_s=0.01 if s < 0 else 300.0, expected_harmful=harm, p_value=0.01 if drop else 0.5,
                             ndcg_drop=drop))
    return pd.DataFrame(rows)


def test_gated_counts_utilization_and_cost_over_every_window():
    table, summary = two_stage.gated(gated_rows(), judge_threshold=0.75, min_drop=0.02)
    f1 = table.loc[("F1_x", 0.5)]
    # the judge ran on 4 of the 8 F1 windows, harmful or not, and cost is their mean
    assert f1.windows == 8 and f1.judged == 0.5 and f1.cost_s == pytest.approx((4 * 300 + 4 * 0.01) / 8)
    assert summary.windows == 17 and summary.judged == pytest.approx(9 / 17)
    assert summary.cost_s == pytest.approx((9 * 300 + 8 * 0.01) / 17)


def test_gated_rates_use_only_windows_the_ground_truth_allows():
    table, summary = two_stage.gated(gated_rows(), judge_threshold=0.75, min_drop=0.02)
    f1 = table.loc[("F1_x", 0.5)]
    assert f1.scored == 4 and f1.alarm == 0.5           # detection over the harmful windows
    h = table.loc[("H_y", 0.0)]
    assert h.windows == 5 and h.scored == 4 and h.alarm == 0.25  # the harmful control window is left out
    assert table.loc[("clean", 0.0)].alarm == 0
    assert summary.detection == 0.5 and summary.false_alarm == 1 / 8
    assert summary.harmful_windows == 4 and summary.harmless_windows == 8


def test_a_judged_window_that_scores_zero_counts_as_judged():
    table, _ = two_stage.gated(gated_rows(), judge_threshold=0.75, min_drop=0.02)
    assert table.loc[("H_y", 0.0)].judged == 3 / 5      # 0.76, 0.0 and the harmful window's 0.90


def test_gated_needs_the_detector():
    with pytest.raises(ValueError, match="no rows"):
        two_stage.gated(gated_rows(), 0.75, detector="other")
