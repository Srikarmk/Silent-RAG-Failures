import pandas as pd
import pytest

from silent_rag.eval.detector_scoring import _auc_pool
from silent_rag.eval.ground_truth import window_has_significant_ndcg10_drop
from silent_rag.eval.window_analyses import (
    _comparisons,
    auroc_statistic_harmful_vs_negative_windows_by_min_ndcg10_drop,
    auroc_statistic_harmful_vs_negative_windows_by_severity,
    check_mean_window_ndcg10_drop_against_reference,
    median_seconds_per_statistic_evaluation,
    one_row_per_window,
    share_of_windows_with_significant_ndcg10_drop_by_severity,
    spearman_statistic_vs_window_mean_ndcg10_drop,
    window_p_value_vs_mean_ndcg10_drop,
)


def table(windows, statistics=("ks_top1", "re_embed")):
    """One row per window x statistic, as the runner writes. windows: (failure, severity, seed, trial, drop, p)."""
    return pd.DataFrame([
        {"failure": f, "severity": s, "seed": seed, "trial": t, "ndcg_drop": d, "p_value": p,
         "detector": stat, "cost_s": 1.0}
        for f, s, seed, t, d, p in windows for stat in statistics])


CLEAN = [("clean", 0.0, seed, t, 0.0, 1.0) for seed in (0, 1) for t in (0, 1)]
F1 = [("F1", 0.5, 0, 0, 0.10, 0.01), ("F1", 0.5, 0, 1, 0.03, 0.01),
      ("F1", 0.5, 1, 0, 0.10, 0.20), ("F1", 0.5, 1, 1, 0.00, 0.04)]


def test_share_counts_windows_not_rows_and_includes_the_baseline():
    got = share_of_windows_with_significant_ndcg10_drop_by_severity(table(CLEAN + F1), [0.0, 0.05], alpha=0.05)
    got = got.set_index(["failure", "min_mean_ndcg10_drop"])
    # two statistics per window: without deduplication n_windows would be 8 and n_significant 6
    assert got.loc[("F1", 0.0), ["n_windows", "n_significant", "share_percent"]].tolist() == [4, 3, 75.0]
    assert got.loc[("F1", 0.05), ["n_windows", "n_significant", "share_percent"]].tolist() == [4, 1, 25.0]
    assert got.loc[("clean", 0.0), ["n_windows", "share_percent"]].tolist() == [4, 0.0]


def test_p_value_table_has_one_row_per_window_and_the_labels_of_the_rule():
    got = window_p_value_vs_mean_ndcg10_drop(table(CLEAN + F1), alpha=0.05, min_mean_ndcg10_drops=[0.0, 0.05])
    assert len(got) == len(CLEAN + F1)  # two statistic rows per window in the input
    for min_drop in (0.0, 0.05):
        expected = window_has_significant_ndcg10_drop(got["p_value"].to_numpy(), got["ndcg_drop"].to_numpy(), 0.05, min_drop)
        assert got[f"significant_at_min_drop_{min_drop}"].tolist() == expected.tolist()
    assert got["significant_at_min_drop_0.0"].sum() == 3 and got["significant_at_min_drop_0.05"].sum() == 1


def test_monte_carlo_flag_on_either_side_of_the_boundary():
    # 2 * sqrt(0.05 * 0.95 / 2000) = 0.009747
    p_values = [0.05 + 0.0097, 0.05 + 0.0098, 0.05 - 0.0097, 0.05 - 0.0098, 0.05]
    windows = [("F1", 0.5, 0, t, 0.1, p) for t, p in enumerate(p_values)]
    got = window_p_value_vs_mean_ndcg10_drop(table(windows), alpha=0.05, min_mean_ndcg10_drops=[0.0])
    assert got.sort_values("trial")["p_value_within_2_monte_carlo_se_of_alpha"].tolist() == [True, False, True, False, True]


def test_p_value_table_records_alpha_and_the_monte_carlo_band():
    got = window_p_value_vs_mean_ndcg10_drop(table(CLEAN + F1), alpha=0.05, min_mean_ndcg10_drops=[0.0])
    half_width = 2 * (0.05 * 0.95 / 2000) ** 0.5
    assert (got["alpha"] == 0.05).all()
    assert got["monte_carlo_band_low"].tolist() == pytest.approx([0.05 - half_width] * len(got))
    assert got["monte_carlo_band_high"].tolist() == pytest.approx([0.05 + half_width] * len(got))
    inside = (got["p_value"] > got["monte_carlo_band_low"]) & (got["p_value"] < got["monte_carlo_band_high"])
    assert got["p_value_within_2_monte_carlo_se_of_alpha"].tolist() == inside.tolist()


def test_rows_of_one_window_must_agree():
    runs = table(F1)
    runs.loc[1, "ndcg_drop"] = 0.5  # second statistic's row of the first window
    with pytest.raises(ValueError, match="disagree"):
        one_row_per_window(runs)


def test_median_seconds_uses_every_row_and_counts_evaluations():
    runs = table(CLEAN)
    runs.loc[runs.detector == "re_embed", "cost_s"] = [1.0, 2.0, 3.0, 10.0]
    runs.loc[runs.detector == "ks_top1", "cost_s"] = 0.1
    got = median_seconds_per_statistic_evaluation(runs).set_index("statistic")
    assert got.loc["re_embed"].tolist() == [2.5, 4]
    assert got.loc["ks_top1"].tolist() == [pytest.approx(0.1), 4]
    assert got.index.tolist() == ["ks_top1", "re_embed"]  # cheapest first


# seeds 0-2 average 0.044 at severity 0.5; seed 3 is far off
REF_RUNS = [("F1", 0.5, seed, 0, d, 0.01) for seed, d in [(0, 0.040), (1, 0.044), (2, 0.048), (3, 0.300)]]
REFERENCE = {"F1": {0.5: 0.044}}


def test_reference_check_uses_only_the_given_seeds():
    got = check_mean_window_ndcg10_drop_against_reference(table(REF_RUNS), REFERENCE, seeds=[0, 1, 2])
    assert got[["n_windows", "expected"]].iloc[0].tolist() == [3, 0.044]
    assert got["observed"].iloc[0] == pytest.approx(0.044)


def test_reference_check_fails_when_other_seeds_are_included():
    with pytest.raises(ValueError, match=r"\(F1, 0.5\): observed 0.10800, expected 0.044"):
        check_mean_window_ndcg10_drop_against_reference(table(REF_RUNS), REFERENCE, seeds=[0, 1, 2, 3])


@pytest.mark.parametrize("reference, seeds, match", [
    ({"F1": {0.9: 0.1}}, [0, 1, 2], r"\(F1, 0.9\): no windows"),
    ({"F2": {0.5: 0.1}}, [0, 1, 2], r"\(F2, 0.5\): no windows"),
    (REFERENCE, [0, 1, 7], r"seeds \[7\] are not in the results table"),
])
def test_reference_check_reports_missing_conditions_and_seeds(reference, seeds, match):
    with pytest.raises(ValueError, match=match):
        check_mean_window_ndcg10_drop_against_reference(table(REF_RUNS), reference, seeds=seeds)


def test_reference_check_filters_trials_like_seeds():
    runs = table(REF_RUNS[:3] + [("F1", 0.5, 0, 1, 0.300, 0.01)])  # trial 1 is far off
    got = check_mean_window_ndcg10_drop_against_reference(runs, REFERENCE, seeds=[0, 1, 2], trials=[0])
    assert got["n_windows"].iloc[0] == 3
    with pytest.raises(ValueError, match=r"trials \[5\] are not in the results table"):
        check_mean_window_ndcg10_drop_against_reference(runs, REFERENCE, seeds=[0, 1, 2], trials=[0, 5])


def scored(windows):
    """ResultRow columns for one statistic. windows: (failure, severity, seed, drop, p, score[, trials]),
    trials (0, 1) unless given, so extra windows of one seed get their own trials."""
    return pd.DataFrame([
        {"run_id": "r", "dataset": "nfcorpus", "model": "m", "model_revision": "v", "window_size": 100,
         "detector": "s", "failure": f, "severity": sev, "seed": seed, "trial": t, "ndcg_drop": d, "p_value": p,
         "score": score, "expected_harmful": f not in ("clean", "H1"), "harmful": False}
        for f, sev, seed, d, p, score, *trials in windows for t in (trials[0] if trials else (0, 1))])


SEEDS = range(4)
BASE = [("clean", 0.0, seed, 0.0, 1.0, 0.0) for seed in SEEDS]


def auroc(windows, min_drop=0.0):
    return auroc_statistic_harmful_vs_negative_windows_by_severity(
        scored(windows), "s", "F1", min_drop, alpha=0.05, resamples=200, seed=0)


def test_auroc_perfect_separation_and_identical_scores():
    got = auroc(BASE + [("F1", 0.5, seed, 0.1, 0.01, 1.0) for seed in SEEDS])
    assert got[["auroc", "ci_low", "ci_high", "n_positive", "n_negative"]].iloc[0].tolist() == [1.0, 1.0, 1.0, 8, 8]
    assert got["negatives"].tolist() == ["unmodified"]  # no negative controls, one definition
    same = [(f, s, seed, d, p, 0.5) for f, s, seed, d, p, _ in BASE + [("F1", 0.5, seed, 0.1, 0.01, 1.0) for seed in SEEDS]]
    assert auroc(same)["auroc"].iloc[0] == 0.5


def test_auroc_without_positives_is_nan_with_a_note():
    got = auroc(BASE + [("F1", 0.1, seed, 0.0, 0.5, 1.0) for seed in SEEDS]).iloc[0]
    assert pd.isna(got["auroc"]) and got["note"] == "undefined: no positive windows"
    assert (got["n_positive"], got["n_modified_windows_excluded"]) == (0, 8)


MIXED = (BASE + [("F1", 0.5, seed, 0.10, 0.01, 1.0) for seed in SEEDS]
         + [("F1", 0.5, seed, 0.03, 0.01, 0.9, (2, 3)) for seed in (0, 1)])
# non-significant H1 windows score 0.2, except seed 3's two at 0.95 (above the 0.9-score positives)
H1 = ([("H1", 0.5, seed, 0.0, 0.5, 0.95 if seed == 3 else 0.2) for seed in SEEDS]
      + [("H1", 0.5, 0, 0.1, 0.01, 0.95, (2, 3))])


def test_auroc_both_negative_definitions_and_excluded_counts():
    got = auroc(MIXED + H1).set_index("negatives")
    cols = ["n_positive", "n_negative", "n_modified_windows_excluded", "n_negative_control_windows_excluded"]
    assert got.loc["unmodified", cols].tolist() == [12, 8, 0, 0]
    assert got.loc["unmodified_and_negative_controls", cols].tolist() == [12, 16, 0, 2]
    # positives: 8 at 1.0, 4 at 0.9. With H1: 16 negatives, two of them (0.95) above the 0.9 positives
    assert got.loc["unmodified", "auroc"] == 1.0
    assert got.loc["unmodified_and_negative_controls", "auroc"] == pytest.approx((8 * 16 + 4 * 14) / (12 * 16))


def test_auroc_by_min_drop_pools_severities():
    windows = MIXED + [("F1", 0.1, seed, 0.0, 0.5, 0.1) for seed in SEEDS]  # severity 0.1: never significant
    got = auroc_statistic_harmful_vs_negative_windows_by_min_ndcg10_drop(
        scored(windows), "s", "F1", [0.0, 0.05], alpha=0.05, resamples=200, seed=0).set_index("min_mean_ndcg10_drop")
    cols = ["n_positive", "n_negative", "n_modified_windows_excluded"]
    assert got.loc[0.0, cols].tolist() == [12, 8, 8]
    assert got.loc[0.05, cols].tolist() == [8, 8, 12]
    assert (got["auroc"] == 1.0).all() and (got["negatives"] == "unmodified").all()


@pytest.mark.parametrize("per_severity", [True, False], ids=["by-severity", "by-min-drop"])
@pytest.mark.parametrize("min_drop", [0.0, 0.05])
def test_counts_match_the_rows_auc_pool_scores(per_severity, min_drop):
    windows = MIXED + H1 + [("F1", 0.1, seed, 0.0, 0.5, 0.1) for seed in SEEDS]
    checked = 0
    for _, _, pool, counts in _comparisons(scored(windows), "s", "F1", 0.05, min_drop, per_severity):
        if counts["n_positive"] and counts["n_negative"]:
            kept = _auc_pool(pool, "s", "F1")
            assert len(kept) == counts["n_positive"] + counts["n_negative"]
            assert kept.groupby(["run_id", "seed"]).ngroups == counts["n_seeds"]
            checked += 1
    assert checked >= 2  # both negatives definitions were compared


def test_auroc_n_seeds_counts_only_seeds_in_the_comparison():
    # seed 4 has only excluded (non-significant) F1 windows and no unmodified windows: not resampled
    windows = BASE + [("F1", 0.5, seed, 0.1, 0.01, 1.0) for seed in (0, 1)] + [("F1", 0.5, 4, 0.0, 0.5, 1.0)]
    got = auroc(windows).iloc[0]
    assert (got["n_seeds"], got["n_modified_windows_excluded"]) == (4, 2)
    by_min_drop = auroc_statistic_harmful_vs_negative_windows_by_min_ndcg10_drop(
        scored(windows), "s", "F1", [0.0], alpha=0.05, resamples=200, seed=0)
    assert by_min_drop["n_seeds"].tolist() == [4]
    no_positives = auroc(BASE + [("F1", 0.5, seed, 0.0, 0.5, 1.0) for seed in SEEDS]).iloc[0]
    assert pd.isna(no_positives["auroc"]) and no_positives["n_seeds"] == 4  # a count, filled even when AUROC is NaN


def spear(windows, resamples=500):
    """windows: (failure, severity, seed, trial, drop, score) for statistic "s"."""
    runs = pd.DataFrame([{"detector": "s", "failure": f, "severity": sev, "seed": seed, "trial": t,
                          "ndcg_drop": d, "score": sc} for f, sev, seed, t, d, sc in windows])
    return spearman_statistic_vs_window_mean_ndcg10_drop(runs, "s", "F1", resamples=resamples, seed=0)


MONOTONE = [("F1", 0.5, seed, t, 0.01 * seed + 0.001 * t, 0.0) for seed in range(3) for t in range(4)]


@pytest.mark.parametrize("sign", [1, -1])
def test_spearman_monotone_and_reversed(sign):
    got = spear([(f, s, seed, t, d, sign * d) for f, s, seed, t, d, _ in MONOTONE])
    assert got["rho"].tolist() == [sign, sign]  # the severity row and the pooled row
    assert got[["n_windows", "n_seeds"]].iloc[0].tolist() == [12, 3]
    assert got["pooled"].tolist() == [False, True] and pd.isna(got["severity"].iloc[1])


def test_spearman_constant_statistic_is_nan_with_a_note():
    got = spear(MONOTONE).iloc[0]
    assert pd.isna(got["rho"]) and got["note"] == "undefined: constant statistic"


def test_spearman_pooled_mixes_severity_with_tracking():
    # within a severity the statistic falls as the drop rises; across severities both rise
    windows = [("F1", sev, seed, t, base + 0.001 * (4 * seed + t), base * 10 - 0.001 * (4 * seed + t))
               for sev, base in [(0.1, 0.01), (0.9, 0.1)] for seed in range(3) for t in range(4)]
    got = spear(windows).set_index("pooled")
    assert got.loc[False, "rho"].tolist() == [-1.0, -1.0]
    assert got.loc[True, "rho"] > 0.4


def test_spearman_excludes_unmodified_windows():
    clean = [("clean", 0.0, seed, t, 0.0, 99.0) for seed in range(3) for t in range(4)]  # would break monotonicity
    got = spear(clean + [(f, s, seed, t, d, d) for f, s, seed, t, d, _ in MONOTONE])
    assert got["rho"].tolist() == [1.0, 1.0] and got["n_windows"].tolist() == [12, 12]


def test_spearman_bootstrap_resamples_whole_seeds():
    # seed 0 tracks the drop (+1), seed 1 is reversed (-1). Resampling whole seeds draws {0,0}, {1,1}
    # or a mix, so the interval reaches exactly -1 and +1; resampling single windows would not.
    up = [("F1", 0.5, 0, t, 0.01 * t, 0.01 * t) for t in range(5)]
    down = [("F1", 0.5, 1, t, 0.01 * t, -0.01 * t) for t in range(5)]
    got = spear(up + down, resamples=2000).iloc[0]
    assert (got["ci_low"], got["ci_high"]) == (-1.0, 1.0)
    assert got["ci_low"] <= got["rho"] <= got["ci_high"]


def test_higher_min_drop_moves_windows_from_positive_to_excluded():
    loose, strict = auroc(MIXED, min_drop=0.0).iloc[0], auroc(MIXED, min_drop=0.05).iloc[0]
    assert (loose["n_positive"], loose["n_modified_windows_excluded"]) == (12, 0)
    assert (strict["n_positive"], strict["n_modified_windows_excluded"]) == (8, 4)  # the 0.03-drop windows
