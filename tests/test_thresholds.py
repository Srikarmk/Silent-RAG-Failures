import numpy as np
import pandas as pd
import pytest

from silent_rag.eval import thresholds
from silent_rag.eval.results import ResultRow, to_frame


def table(seeds=range(10), trials=20, dataset="nfcorpus", model="m", window_size=100):
    """Clean windows score N(0, 1) for `det` and exactly 0 for `zero`; a harmless control
    shifts `det` by +5, and a harmful failure shifts it by +5 on its harmful windows."""
    rng = np.random.default_rng(0)
    rows = []
    for seed in seeds:
        for failure, harmful_flag, shift in (("clean", False, 0.0), ("H_control", False, 5.0), ("F_fail", True, 5.0)):
            for trial in range(trials):
                harmful = harmful_flag and trial % 2 == 0
                base = rng.normal()
                for det, score in (("det", base + (shift if failure == "H_control" or harmful else 0.0)),
                                   ("zero", 0.0)):
                    rows.append(ResultRow(
                        run_id="r", dataset=dataset, model=model, model_revision="rev",
                        failure=failure, expected_harmful=harmful_flag, severity=0.0 if failure == "clean" else 0.5,
                        seed=seed, trial=trial, window_size=window_size, detector=det, score=score,
                        ndcg_drop=0.05 if harmful else 0.0, harmful=harmful, p_value=0.01 if harmful else 0.9,
                        cost_s=0.0))
    return to_frame(rows)


def test_threshold_is_the_clean_quantile_of_the_calibration_seeds_only():
    df = table()
    cal = thresholds.calibrate(df, range(7))
    clean = df[(df.failure == "clean") & (df.detector == "det") & (df.seed < 7)].score
    assert cal["detectors"]["det"]["threshold"] == pytest.approx(np.quantile(clean, 0.95))
    assert cal["detectors"]["det"]["clean_windows"] == 7 * 20
    # moving every failure and control window doesn't change it, nor do the held-out seeds
    moved = df.copy()
    moved.loc[(moved.failure != "clean") | (moved.seed >= 7), "score"] += 100
    assert thresholds.calibrate(moved, range(7))["detectors"] == cal["detectors"]


def test_records_the_setup_and_a_detector_that_scores_zero_never_alarms_on_clean():
    df = table()
    cal = thresholds.calibrate(df, range(7))
    assert {k: cal[k] for k in ("dataset", "model", "model_revision", "window_size", "run_id")} == {
        "dataset": "nfcorpus", "model": "m", "model_revision": "rev", "window_size": 100, "run_id": "r"}
    assert cal["calibration_seeds"] == list(range(7)) and cal["false_alarm"] == 0.05
    assert cal["detectors"]["zero"]["threshold"] == 0.0
    rates = thresholds.alarm_rates(df, {"zero": 0.0}, exclude_seeds=range(7))
    assert (rates["zero"] == 0).all()


def test_alarm_rates_on_held_out_seeds():
    df = table()
    t = {n: d["threshold"] for n, d in thresholds.calibrate(df, range(7))["detectors"].items()}
    rates = thresholds.alarm_rates(df, t, exclude_seeds=range(7))
    assert rates.loc[("clean", 0.0), "det"] < 0.15  # ~5% on 60 held-out windows
    assert rates.loc[("H_control", 0.5), "det"] > 0.9  # a control that moves the score is a false alarm
    assert rates.loc[("F_fail", 0.5), "det"] > 0.9  # harmful windows only


@pytest.mark.parametrize("change, message", [
    (lambda df: df.assign(dataset="scifact"), "NFCorpus only"),
    (lambda df: df.assign(model=np.where(df.seed == 0, "a", "b")), "mix model"),
    (lambda df: df.assign(window_size=np.where(df.seed == 0, 50, 100)), "mix window_size"),
])
def test_refuses_held_out_datasets_and_mixed_setups(change, message):
    with pytest.raises(ValueError, match=message):
        thresholds.calibrate(change(table()), range(7))


def test_seed_checks():
    with pytest.raises(ValueError, match="no windows for calibration seeds"):
        thresholds.calibrate(table(seeds=range(5)), range(7))
    with pytest.raises(ValueError, match="held out"):
        thresholds.calibrate(table(seeds=range(7)), range(7))
    with pytest.raises(ValueError, match="false_alarm"):
        thresholds.calibrate(table(), range(7), false_alarm=0.0)


def test_duplicate_clean_windows_are_an_error():
    df = table()
    with pytest.raises(ValueError, match="twice"):
        thresholds.calibrate(pd.concat([df, df.iloc[:1]]), range(7))


def test_save_and_load_round_trip_and_setup_check(tmp_path):
    cal = thresholds.calibrate(table(), range(7))
    path = tmp_path / "thresholds.json"
    thresholds.save(path, cal)
    assert "silent_rag.eval.thresholds" in path.read_text()
    loaded = thresholds.load(path, model="m", model_revision="rev", window_size=100)
    assert loaded == {n: d["threshold"] for n, d in cal["detectors"].items()}
    with pytest.raises(ValueError, match="window_size"):
        thresholds.load(path, window_size=50)
    with pytest.raises(ValueError, match="model"):
        thresholds.load(path, model="other")


def test_leave_one_seed_out_sees_a_seed_with_an_unlucky_reference():
    df = table()
    steady = thresholds.leave_one_seed_out(df)
    assert steady.loc["det", "mean"] < 0.15 and steady.loc["zero", "worst_seed"] == 0.0
    # one seed whose reference makes every clean window score higher
    shifted = df.copy()
    shifted.loc[(shifted.seed == 9) & (shifted.detector == "det"), "score"] += 3
    rates = thresholds.leave_one_seed_out(shifted)
    assert rates.loc["det", "worst_seed"] > 0.8


def test_a_control_window_that_did_harm_is_not_a_false_alarm():
    df = table()
    t = {"det": 0.0}
    # every control window alarms (scores are +5); mark half of them as having done harm
    bad = (df.failure == "H_control") & (df.trial % 2 == 0)
    df.loc[bad, ["p_value", "ndcg_drop"]] = [0.01, 0.05]
    rates = thresholds.alarm_rates(df, t)
    counted = df[(df.failure == "H_control") & (df.detector == "det") & ~bad]
    assert rates.loc[("H_control", 0.5), "det"] == pytest.approx((counted.score > 0).mean())
    # with a min_drop above their drop, those windows are harmless again and count
    assert thresholds.alarm_rates(df, t, min_drop=0.1).loc[("H_control", 0.5), "det"] == pytest.approx(
        (df[(df.failure == "H_control") & (df.detector == "det")].score > 0).mean())


def test_harmful_is_recomputed_with_min_drop():
    df = table()
    t = {n: d["threshold"] for n, d in thresholds.calibrate(df, range(7))["detectors"].items()}
    # the failure's harmful windows drop 0.05: above min_drop 0.1 none are harmful, so none are kept
    assert ("F_fail", 0.5) in thresholds.alarm_rates(df, t).index
    assert ("F_fail", 0.5) not in thresholds.alarm_rates(df, t, min_drop=0.1).index
