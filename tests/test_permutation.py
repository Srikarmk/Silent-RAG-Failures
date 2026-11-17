"""Per-window permutation tests for the FeatureDrift detectors (#73). Offline."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY
from silent_rag.detect.features import FEATURES, Permutation, permutation_p_value
from silent_rag.detect.ks import KS_DETECTORS, KS_PERMUTATION, ks_statistic
from silent_rag.detect.psi import PSI_DETECTORS, PSI_PERMUTATION
from silent_rag.detect.qpp import NQC, QPP_PERMUTATION, MaxScore
from silent_rag.eval.permutation import compare_threshold_rules
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.pipeline import Retriever
from test_ks import NoisyDocs, obs
from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset

BASE = [f"{p}_{f}" for p in ("ks", "psi") for f in FEATURES] + ["qpp_max", "qpp_gap", "qpp_nqc"]


def test_every_feature_drift_detector_has_a_permutation_variant():
    assert len(BASE) == 2 * len(FEATURES) + 3
    for name in BASE:
        base, perm = REGISTRY[name], REGISTRY[f"{name}_perm"]
        assert issubclass(perm, Permutation) and issubclass(perm, base)
        assert perm.feature == base.feature
        assert not issubclass(base, Permutation)  # the plain detectors are untouched
    assert len([n for n in REGISTRY if n.endswith("_perm")]) == len(BASE)


def test_plain_scores_are_unchanged():
    rng = np.random.default_rng(0)
    reference, window = rng.normal(0.6, 0.05, 100), rng.normal(0.58, 0.05, 100)
    ks = KS_DETECTORS["top1"]()
    ks.fit(obs(reference))
    # float32 scores in the Observation, so compare against scipy on what the detector saw
    assert ks.score(obs(window)) == ks_2samp(ks.reference_, ks.values(obs(window)), method="asymp").statistic
    qpp = MaxScore()
    qpp.fit(obs(reference))
    assert qpp.score(obs(window)) == pytest.approx(qpp.reference_.mean() - qpp.values(obs(window)).mean())


@pytest.mark.parametrize("n_ref, n_cur, decimals", [(100, 100, None), (30, 70, None), (50, 50, 1)])
def test_ks_statistic_matches_scipy(n_ref, n_cur, decimals):
    rng = np.random.default_rng(n_ref + n_cur)
    a, b = rng.normal(size=n_ref), rng.normal(0.3, 1, size=n_cur)
    if decimals is not None:  # heavy ties
        a, b = a.round(decimals), b.round(decimals)
    assert ks_statistic(a, b) == pytest.approx(ks_2samp(a, b, method="asymp").statistic, abs=1e-12)


def test_statistic_equals_the_fitted_score_on_the_real_split():
    rng = np.random.default_rng(1)
    reference, window = obs(rng.normal(0.6, 0.05, 100)), obs(rng.normal(0.55, 0.06, 100), seed=1)
    for name in BASE:
        det = REGISTRY[name]()
        det.fit(reference)
        assert det.statistic(det.reference_, det.values(window)) == pytest.approx(det.score(window), abs=1e-12), name


def test_psi_statistic_refits_its_bins_on_the_reference_it_is_given():
    rng = np.random.default_rng(2)
    a, b, c = (rng.normal(m, 0.05, 100) for m in (0.6, 0.4, 0.45))
    fitted_on_a = PSI_DETECTORS["top1"]()
    fitted_on_a.fit(obs(a))
    fitted_on_b = PSI_DETECTORS["top1"]()
    fitted_on_b.fit(obs(b))
    # asking the detector fitted on `a` about (b, c) must not use a's bins
    assert fitted_on_a.statistic(fitted_on_b.reference_, fitted_on_b.values(obs(c))) == pytest.approx(
        fitted_on_b.score(obs(c)), abs=1e-12)
    assert fitted_on_a.score(obs(c)) != pytest.approx(fitted_on_b.score(obs(c)))


def mean_shift(reference, current):
    return current.mean() - reference.mean()


def test_p_value_by_hand():
    # 3 vs 3, all separated: 1 of the 20 splits is as extreme as the observed one
    reference, current = np.zeros(3), np.ones(3)
    p = permutation_p_value(mean_shift, reference, current, 9999, np.random.default_rng(0))
    assert p == pytest.approx(1 / 20, abs=0.01)
    # the opposite direction is the least extreme split, so every reshuffle counts
    assert permutation_p_value(mean_shift, current, reference, 199, np.random.default_rng(0)) == 1.0
    # identical values: every reshuffle ties with the observed statistic
    assert permutation_p_value(mean_shift, np.ones(5), np.ones(5), 99, np.random.default_rng(0)) == 1.0


def test_p_value_is_bounded_deterministic_and_never_zero():
    rng = np.random.default_rng(3)
    reference = obs(rng.normal(0.6, 0.05, 100))
    shifted, same = obs(rng.normal(0.4, 0.05, 100), seed=1), obs(rng.normal(0.6, 0.05, 100), seed=2)
    for perm in (KS_PERMUTATION["top1"], PSI_PERMUTATION["top1"], QPP_PERMUTATION["qpp_max"]):
        det = perm(permutations=199)
        det.fit(reference)
        assert det.p_value(shifted) == 1 / 200  # the floor, not 0
        assert det.score(shifted) == 1 - 1 / 200
        p = det.p_value(same)
        assert 1 / 200 <= p <= 1 and det.p_value(same) == p  # same reshuffles every call
        assert det.score(same) == 1 - p


def test_seed_changes_the_reshuffles_not_the_conclusion():
    rng = np.random.default_rng(4)
    reference, window = obs(rng.normal(0.6, 0.05, 100)), obs(rng.normal(0.59, 0.05, 100), seed=1)
    ps = []
    for seed in (0, 1):
        det = KS_PERMUTATION["top1"](permutations=499, seed=seed)
        det.fit(reference)
        ps.append(det.p_value(window))
    assert ps[0] != ps[1] and abs(ps[0] - ps[1]) < 0.1


@pytest.mark.parametrize("name", ["ks_top1", "psi_top1", "qpp_max"])
def test_null_is_calibrated_for_disjoint_windows_of_one_pool(name):
    """The runner's clean case: reference and window are disjoint random subsets of one
    query pool. About alpha of them should come out below alpha, PSI included."""
    rng = np.random.default_rng(5)
    det = REGISTRY[name]()
    pool = rng.normal(0.6, 0.05, 323)
    rejections, n = 0, 200
    for _ in range(n):
        draw = rng.permutation(pool)
        p = permutation_p_value(det.statistic, draw[:100], draw[100:200], 99, rng)
        rejections += p <= 0.05
    assert 0.01 <= rejections / n <= 0.10, rejections / n


def test_psi_with_the_fitted_bins_would_not_be_calibrated():
    """Why PSI.statistic refits: reshuffling against the bins from fit() rejects far too often."""
    rng = np.random.default_rng(6)
    rejections, n = 0, 100
    for _ in range(n):
        det = PSI_DETECTORS["top1"]()
        reference, window = rng.normal(0.6, 0.05, 100), rng.normal(0.6, 0.05, 100)
        det.fit(obs(reference))
        fixed_bins = lambda ref, cur: det.compare(ref, cur)  # noqa: E731, ignores `ref`
        rejections += permutation_p_value(fixed_bins, det.reference_, det.values(obs(window)), 99, rng) <= 0.05
    assert rejections / n > 0.3


def test_qpp_is_one_sided():
    rng = np.random.default_rng(7)
    reference = obs(rng.normal(0.6, 0.05, 100))
    det = QPP_PERMUTATION["qpp_max"](permutations=199)
    det.fit(reference)
    less_confident, more_confident = obs(rng.normal(0.5, 0.05, 100), seed=1), obs(rng.normal(0.7, 0.05, 100), seed=2)
    assert det.p_value(less_confident) == 1 / 200
    assert det.p_value(more_confident) == 1.0  # looking better is not an alarm
    assert det.score(more_confident) == 0.0


def test_parameters_are_checked_and_passed_through():
    for bad in (0, -1, 1.5, True, "999"):
        with pytest.raises(ValueError, match="permutations"):
            KS_PERMUTATION["top1"](permutations=bad)
    with pytest.raises(ValueError, match="seed"):
        KS_PERMUTATION["top1"](seed=-1)
    with pytest.raises(TypeError):
        KS_PERMUTATION["top1"](bins=5)  # KS has no bins
    assert KS_PERMUTATION["top1"]().permutations == 999
    psi = PSI_PERMUTATION["top1"](permutations=50, bins=5)
    assert (psi.permutations, psi.bins) == (50, 5)
    with pytest.raises(ValueError, match="bins"):
        PSI_PERMUTATION["top1"](bins=1)
    nqc = QPP_PERMUTATION["qpp_nqc"](permutations=50)
    assert nqc.depth == NQC().depth
    with pytest.raises(ValueError, match="score_depth"):
        QPP_PERMUTATION["qpp_nqc"].check_config({"depth": 100, "permutations": 50}, {})
    with pytest.raises(RuntimeError, match="fit"):
        KS_PERMUTATION["top1"]().score(obs(np.full(10, 0.5)))


def toy_run(monkeypatch, detectors, seeds=(0,)):
    for inj in (ScrambleDocs, NoisyDocs):
        monkeypatch.setitem(INJECTORS, inj.name, inj)
    cfg = {
        "datasets": ["nfcorpus"], "failures": {"test_noisy_docs": [0.5]}, "detectors": detectors,
        "window_size": 15, "trials": 4, "seeds": list(seeds),
    }
    return runner.run(cfg, loader=lambda name: toy_dataset(),
                      make_retriever=lambda: Retriever("fake", encoder=HashEncoder()))


def test_all_variants_run_through_the_runner_next_to_the_plain_detectors(monkeypatch):
    perm = [{f"{name}_perm": {"permutations": 49}} for name in BASE]
    df = toy_run(monkeypatch, BASE + perm)
    assert set(df.detector) == set(BASE) | {f"{n}_perm" for n in BASE}
    p = 1 - df[df.detector.str.endswith("_perm")].score
    assert ((p >= 1 / 50 - 1e-12) & (p <= 1 + 1e-12)).all()
    assert (df.cost_s > 0).all() and np.isfinite(df.cost_s).all()  # the runner times them like any detector
    # adding the variants does not move the plain detectors' scores
    alone = toy_run(monkeypatch, BASE)
    key = ["failure", "severity", "seed", "trial", "detector"]
    both = df[df.detector.isin(BASE)].sort_values(key).score.to_numpy()
    np.testing.assert_array_equal(both, alone.sort_values(key).score.to_numpy())


def test_bad_permutations_fail_before_the_run_starts(monkeypatch):
    with pytest.raises(SystemExit, match="ks_top1_perm"):
        toy_run(monkeypatch, [{"ks_top1_perm": {"permutations": 0}}])


def results(rows):
    """rows: (detector, failure, severity, expected_harmful, seed, trial, score, ndcg_drop, p_value)"""
    cols = ["detector", "failure", "severity", "expected_harmful", "seed", "trial", "score", "ndcg_drop", "p_value"]
    df = pd.DataFrame(rows, columns=cols)
    df["run_id"], df["dataset"] = "run", "nfcorpus"
    return df


def hand_table():
    rows = []

    def window(failure, severity, harmful, seed, trial, stat, p_perm, drop=0.0, p=1.0):
        rows.append(("ks_top1", failure, severity, harmful, seed, trial, stat, drop, p))
        rows.append(("ks_top1_perm", failure, severity, harmful, seed, trial, 1 - p_perm, drop, p))

    # calibration seed 0: clean statistics 0.00 ... 0.19, so the 95% quantile is 0.1805
    for t in range(20):
        window("clean", 0.0, False, 0, t, t / 100, 0.5)
    # evaluation seeds 1 and 2
    for seed in (1, 2):
        for t in range(4):
            # clean: the clean rule fires on 1 of 4 in seed 1 only, the permutation rule on 2 of 4 in both
            window("clean", 0.0, False, seed, t, 0.30 if (seed == 1 and t == 0) else 0.05, 0.01 if t < 2 else 0.6)
            # a control that changes the scores without harm: both rules fire on all of it
            window("H_swap", 1.0, False, seed, t, 0.9, 0.001, drop=-0.02)
            # a control at another severity that neither rule sees
            window("H_swap", 0.5, False, seed, t, 0.05, 0.4)
            # a failure: harmful in trials 0-1 only; the clean rule catches both, the permutation rule one
            window("F1", 0.5, True, seed, t, 0.5 if t < 2 else 0.0, 0.01 if t == 0 else 0.2,
                   drop=0.05 if t < 2 else 0.001, p=0.01 if t < 2 else 0.4)
    return results(rows)


def test_compare_threshold_rules_by_hand():
    out = compare_threshold_rules(hand_table(), calibration_seeds=[0])
    assert list(out.kind) == ["clean", "control", "control", "failure"]
    by = {(r.failure, r.severity): r for r in out.itertuples()}
    clean = by[("clean", 0.0)]
    assert clean.clean_threshold == pytest.approx(0.1805)
    assert (clean.n_windows, clean.n_seeds) == (8, 2)  # calibration windows are not evaluated
    assert clean.rate_clean_rule == pytest.approx(1 / 8) and clean.rate_permutation == pytest.approx(0.5)
    assert clean.seed_range_clean_rule == (0.0, 0.25) and clean.seed_range_permutation == (0.5, 0.5)
    assert clean.both_alarm == pytest.approx(1 / 8)
    # each control severity is its own row
    assert (by[("H_swap", 1.0)].rate_clean_rule, by[("H_swap", 1.0)].rate_permutation) == (1.0, 1.0)
    assert (by[("H_swap", 0.5)].rate_clean_rule, by[("H_swap", 0.5)].rate_permutation) == (0.0, 0.0)
    # the failure row is detection on its harmful windows only
    f1 = by[("F1", 0.5)]
    assert f1.n_windows == 4 and f1.rate_clean_rule == 1.0 and f1.rate_permutation == 0.5


def test_compare_threshold_rules_rethresholds_harmful_and_checks_its_inputs():
    table = hand_table()
    # at min_drop 0.06 no F1 window is harmful any more, so there is no failure row
    assert "failure" not in set(compare_threshold_rules(table, [0], min_drop=0.06).kind)
    with pytest.raises(ValueError, match="none are left"):
        compare_threshold_rules(table, [0, 1, 2])
    with pytest.raises(ValueError, match="non-empty subset"):
        compare_threshold_rules(table, [])
    with pytest.raises(ValueError, match="non-empty subset"):
        compare_threshold_rules(table, [9])
    with pytest.raises(ValueError, match="no rows for"):
        compare_threshold_rules(table, [0], detectors=["psi_top1"])
    with pytest.raises(ValueError, match="_perm partner"):
        compare_threshold_rules(table[table.detector == "ks_top1"], [0])
    with pytest.raises(ValueError, match="same windows"):
        compare_threshold_rules(table.drop(index=1), [0])
    no_clean = table.copy()
    no_clean.loc[(no_clean.failure == "clean") & (no_clean.seed == 0), "failure"] = "H_other"
    with pytest.raises(ValueError, match="no clean windows"):
        compare_threshold_rules(no_clean, [0])


def test_control_windows_marked_harmful_are_not_false_alarms():
    """Same definition as detector_scoring.false_alarm_rate: alarms among controls the ground
    truth marks harmless. A harmful control window leaves the denominator and is counted."""
    table = hand_table()
    assert (compare_threshold_rules(table, [0]).n_harmful_excluded == 0).all()
    # one H_swap@0.5 window in seed 1 that both rules alarm on and that did measurable harm
    rows = (table.failure == "H_swap") & (table.severity == 0.5) & (table.seed == 1) & (table.trial == 0)
    table.loc[rows & (table.detector == "ks_top1"), "score"] = 0.9
    table.loc[rows & (table.detector == "ks_top1_perm"), "score"] = 0.999
    table.loc[rows, ["p_value", "ndcg_drop"]] = [0.01, 0.03]
    by = {(r.failure, r.severity): r for r in compare_threshold_rules(table, [0]).itertuples()}
    swap = by[("H_swap", 0.5)]
    assert (swap.n_windows, swap.n_harmful_excluded) == (7, 1)
    assert swap.rate_clean_rule == 0.0 and swap.rate_permutation == 0.0  # 1/8 if it were counted
    assert swap.seed_range_permutation == (0.0, 0.0)
    assert by[("clean", 0.0)].n_harmful_excluded == 0 and by[("H_swap", 1.0)].n_harmful_excluded == 0
    # harmful is re-thresholded with min_drop: a 0.03 drop under min_drop 0.05 is harmless, so it counts again
    swap = {(r.failure, r.severity): r for r in compare_threshold_rules(table, [0], min_drop=0.05).itertuples()}[
        ("H_swap", 0.5)]
    assert (swap.n_windows, swap.n_harmful_excluded) == (8, 0)
    assert swap.rate_clean_rule == pytest.approx(1 / 8) and swap.rate_permutation == pytest.approx(1 / 8)


def test_alarm_alpha_does_not_change_harmful_labels():
    """alpha sets the detectors' alarms only; which windows are harmful is set by harm_alpha
    and min_drop. Before they were split, alpha = 0.005 made the F1 windows (bootstrap p 0.01)
    harmless and alpha = 0.5 made the rest (p 0.4) harmful."""
    table = hand_table()
    # one H_swap@0.5 window that did measurable harm at bootstrap p 0.03
    rows = (table.failure == "H_swap") & (table.severity == 0.5) & (table.seed == 1) & (table.trial == 0)
    table.loc[rows, ["p_value", "ndcg_drop"]] = [0.03, 0.03]

    def labels(**kw):
        out = compare_threshold_rules(table, [0], **kw)
        return out[["kind", "failure", "severity", "n_windows", "n_harmful_excluded"]]

    reference = labels()
    # rows: clean, H_swap@0.5, H_swap@1.0, F1@0.5
    assert list(reference.n_windows) == [8, 7, 8, 4] and list(reference.n_harmful_excluded) == [0, 1, 0, 0]
    for alpha in (0.005, 0.02, 0.5):
        pd.testing.assert_frame_equal(labels(alpha=alpha), reference)
        pd.testing.assert_frame_equal(labels(alpha=alpha, min_drop=0.02), labels(min_drop=0.02))
    # harm_alpha is the threshold that moves them: at 0.005 nothing is harmful
    strict = labels(harm_alpha=0.005)
    assert "failure" not in set(strict.kind) and (strict.n_harmful_excluded == 0).all()
    with pytest.raises(ValueError, match="harm_alpha"):
        compare_threshold_rules(table, [0], harm_alpha=0.0)


def test_compare_threshold_rules_on_a_runner_table(monkeypatch):
    df = toy_run(monkeypatch, ["ks_top1", {"ks_top1_perm": {"permutations": 49}}], seeds=(0, 1, 2))
    out = compare_threshold_rules(df, calibration_seeds=[0])
    assert set(out.detector) == {"ks_top1"} and "clean" in set(out.kind)
    assert out[["rate_clean_rule", "rate_permutation"]].apply(lambda c: c.between(0, 1)).all().all()
    assert set(out[out.kind == "clean"].n_seeds) == {2}
