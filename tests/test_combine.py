import json

import numpy as np
import pandas as pd
import pytest

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.detect.combined import CombinedScores
from silent_rag.eval import combine
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.pipeline import Retriever
from silent_rag.run import load_plugins
from test_pipeline import HashEncoder, ScrambleDocs, toy_dataset

TWO = ["ks_top1", "psi_gap"]


def results(seeds=range(6), controls=True):
    """Synthetic runner rows. The failure raises ks_top1 a lot and psi_gap a little; the
    control raises ks_top1 as much and lowers psi_gap. Neither detector alone separates the
    failure from both clean and control windows, the pair does."""
    rng = np.random.default_rng(0)
    rows = []
    conditions = [("clean", 0.0, False, (0, 0)), ("F_x", 0.5, True, (2, 1))]
    if controls:
        conditions.append(("H_x", 1.0, False, (2, -1)))
    for seed in seeds:
        for failure, severity, harmful, shift in conditions:
            for trial in range(20):
                drop = 0.05 if harmful else 0.0
                for det, s in zip(TWO, shift):
                    rows.append({"run_id": "r", "dataset": "nfcorpus", "seed": seed, "failure": failure,
                                 "severity": severity, "trial": trial, "detector": det,
                                 "score": s + rng.normal(), "expected_harmful": harmful,
                                 "p_value": 0.001 if harmful else 0.5, "ndcg_drop": drop})
    return pd.DataFrame(rows)


def test_window_table_has_one_row_per_window_and_rethresholds_harmful():
    wide = combine.window_table(results(), TWO)
    assert len(wide) == 6 * 3 * 20 and set(TWO) <= set(wide.columns)
    assert wide[wide.failure == "F_x"].harmful.all()
    strict = combine.window_table(results(), TWO, min_drop=0.06)
    assert not strict.harmful.any()


def test_window_table_refuses_a_missing_detector():
    with pytest.raises(ValueError, match=r"no rows for \['qpp_max'\]"):
        combine.window_table(results(), TWO + ["qpp_max"])


def test_fit_refuses_training_without_negative_controls():
    with pytest.raises(ValueError, match="no negative-control windows"):
        combine.fit(combine.window_table(results(controls=False), TWO), TWO)


def test_combination_beats_either_detector_when_a_control_fools_one():
    wide = combine.window_table(results(), TWO)
    table = combine.evaluate(wide, train_seeds=[0, 1, 2, 3], detectors=TWO)
    row = table.loc["F_x"]
    assert row.held_out_seeds > row.best_single_auroc + 0.05
    weights = combine.fit(wide[wide.seed < 4], TWO)
    assert weights["coef"][1] > 0  # psi_gap, the detector the control doesn't move
    assert weights["trained_on"]["failures"] == ["F_x", "H_x", "clean"]


def test_false_alarms_are_lower_for_the_combination_on_the_control():
    wide = combine.window_table(results(), TWO)
    train, test = wide[wide.seed < 4], wide[wide.seed >= 4]
    both = combine.false_alarms(combine.fit(train, TWO), test, train)
    alone = combine.false_alarms(combine.fit(train, ["ks_top1"]), test, train)
    assert both.loc[("H_x", 1.0)] < alone.loc[("H_x", 1.0)]


# the detector


def weights_file(tmp_path, **override):
    w = {"detectors": ["ks_top1", "qpp_max"], "mean": [0.1, 0.0], "scale": [0.2, 0.05], "coef": [1.5, -0.5],
         "intercept": -0.3, "trained_on": {"datasets": ["nfcorpus"]}, **override}
    path = tmp_path / "w.json"
    path.write_text(json.dumps(w))
    return str(path), w


def test_detector_applies_the_saved_weights_to_its_parts_scores(tmp_path, monkeypatch):
    load_plugins()
    monkeypatch.setitem(INJECTORS, ScrambleDocs.name, ScrambleDocs)
    path, w = weights_file(tmp_path)
    cfg = {"datasets": ["nfcorpus"], "window_size": 10, "trials": 3, "failures": {ScrambleDocs.name: [0.5]},
           "detectors": ["ks_top1", "qpp_max", {"combined_scores": {"weights": path}}]}
    df = runner.run(cfg, loader=lambda name: toy_dataset(), make_retriever=lambda: Retriever("fake", encoder=HashEncoder()))
    wide = df.pivot_table(index=["failure", "trial"], columns="detector", values="score")
    expected = ((wide[w["detectors"]] - w["mean"]) / w["scale"]) @ np.array(w["coef"]) + w["intercept"]
    np.testing.assert_allclose(wide["combined_scores"], expected, rtol=1e-9)


def test_check_config_accepts_a_good_weights_file(tmp_path):
    load_plugins()
    path, _ = weights_file(tmp_path)
    assert CombinedScores.check_config({"weights": path}, {}) is None


@pytest.mark.parametrize("override, message", [
    ({"trained_on": {"datasets": ["scifact"]}}, "trained on held-out data"),
    ({"detectors": ["ks_top1", "nope"]}, "aren't registered"),
    ({"coef": [1.0]}, "isn't a weights file"),
])
def test_check_config_refuses_bad_weights(tmp_path, override, message):
    load_plugins()
    path, _ = weights_file(tmp_path, **override)
    with pytest.raises(ValueError, match=message):
        CombinedScores.check_config({"weights": path}, {})


def test_check_config_refuses_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="can't read combined_scores weights"):
        CombinedScores.check_config({"weights": str(tmp_path / "none.json")}, {})


def test_registered():
    load_plugins()
    assert DETECTORS["combined_scores"] is CombinedScores
