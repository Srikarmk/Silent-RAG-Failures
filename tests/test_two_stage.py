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
