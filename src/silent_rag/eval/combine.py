"""Combine the cheap score detectors into one (#75).

Each cheap detector scores a window alone. This fits a logistic regression over all
of their scores at once, on NFCorpus results, and saves the weights for the
`combined_scores` detector. The detector itself stays label-free: at run time it
computes the same scores and applies the saved weights.

Training labels follow compute_auroc: positives are harmful windows of harmful
failures; negatives are clean windows and harmless-control windows that stayed
harmless. Damaged-but-not-significant windows are left out, and so are controls
that turned out harmful. The negative controls are what stop the combination from
learning "something changed", so a fit without any is refused.

    python -m silent_rag.eval.combine --results results/dev_full.parquet \
        --train-seeds 0,1,2,3,4,5,6 --out configs/combined_weights.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# the score-based detectors: each reads only logged scores, ~1 ms per check
CHEAP = ["ks_top1", "ks_gap", "ks_mean10", "ks_std10", "psi_top1", "psi_gap", "psi_mean10", "psi_std10",
         "qpp_max", "qpp_nqc", "qpp_gap"]
WINDOW = ["run_id", "dataset", "seed", "failure", "severity", "trial"]


def window_table(df: pd.DataFrame, detectors=CHEAP, alpha: float = 0.05, min_drop: float = 0.0) -> pd.DataFrame:
    """One row per window: its label inputs and one column per detector score.
    `harmful` is recomputed from p_value and ndcg_drop, so any min_drop can be used."""
    missing = sorted(set(detectors) - set(df.detector))
    if missing:
        raise ValueError(f"results have no rows for {missing}")
    rows = df[df.detector.isin(detectors)]
    wide = rows.pivot_table(index=WINDOW, columns="detector", values="score").reset_index()
    meta = rows.drop_duplicates(WINDOW)[WINDOW + ["expected_harmful", "p_value", "ndcg_drop"]]
    wide = wide.merge(meta, on=WINDOW, validate="one_to_one")
    if wide[list(detectors)].isna().any().any():
        raise ValueError("some windows are missing a detector score")
    wide["harmful"] = (wide.p_value < alpha) & (wide.ndcg_drop >= min_drop)
    return wide


def labelled(wide: pd.DataFrame) -> pd.DataFrame:
    """The windows a combination trains and is scored on, with y = 1 for harmful."""
    clean = wide.failure.eq("clean")
    positive = wide.expected_harmful & wide.harmful
    negative = clean | (~wide.expected_harmful & ~clean & ~wide.harmful)
    out = wide[positive | negative].copy()
    out["y"] = positive[positive | negative].astype(int)
    return out


def fit(wide: pd.DataFrame, detectors=CHEAP, C: float = 1.0) -> dict:
    data = labelled(wide)
    controls = data[~data.expected_harmful & ~data.failure.eq("clean")]
    if controls.empty:
        raise ValueError("no negative-control windows to train on; the combination would learn 'any change'")
    if data.y.nunique() != 2:
        raise ValueError("training needs both harmful and harmless windows")
    x = data[list(detectors)].to_numpy(float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    # balanced: there are far more harmful windows than clean ones in a dose grid
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit((x - mean) / scale, data.y)
    return {
        "detectors": list(detectors),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "trained_on": {
            "run_ids": sorted(data.run_id.unique().tolist()),
            "datasets": sorted(data.dataset.unique().tolist()),
            "seeds": sorted(int(s) for s in data.seed.unique()),
            "failures": sorted(data.failure.unique().tolist()),
            "windows": int(len(data)),
            "C": C,
        },
    }


def predict(weights: dict, wide: pd.DataFrame) -> np.ndarray:
    x = wide[weights["detectors"]].to_numpy(float)
    return ((x - np.array(weights["mean"])) / np.array(weights["scale"])) @ np.array(weights["coef"]) + weights["intercept"]


def auroc(weights: dict, wide: pd.DataFrame, failure: str) -> float:
    """Same pool as compute_auroc: one failure's harmful windows against clean and harmless controls."""
    data = labelled(wide)
    pool = data[data.failure.eq(failure) | ~data.expected_harmful]
    if pool.y.nunique() != 2:
        return float("nan")
    return float(roc_auc_score(pool.y, predict(weights, pool)))


def evaluate(wide: pd.DataFrame, train_seeds, detectors=CHEAP, C: float = 1.0) -> pd.DataFrame:
    """Two checks, per harmful failure:
    - held_out_seeds: trained on train_seeds, scored on the other seeds
    - unseen_failure: trained on train_seeds without that failure, scored on the other
      seeds, so the failure is new to it. The fairer test for a failure we haven't built.
    Each single detector's AUROC on the same held-out windows is the baseline."""
    train, test = wide[wide.seed.isin(train_seeds)], wide[~wide.seed.isin(train_seeds)]
    if test.empty or train.empty:
        raise ValueError("train_seeds has to leave some seeds for testing, and use some")
    weights = fit(train, detectors, C)
    rows = []
    for failure in sorted(test[test.expected_harmful].failure.unique()):
        rest = train[~train.failure.eq(failure)]
        # with only one harmful failure there is nothing left to learn "harmful" from
        unseen = (auroc(fit(rest, detectors, C), test, failure)
                  if labelled(rest).y.nunique() == 2 else float("nan"))
        single = {d: auroc({"detectors": [d], "mean": [0.0], "scale": [1.0], "coef": [1.0], "intercept": 0.0},
                           test, failure) for d in detectors}
        best = max(single, key=lambda d: -1 if np.isnan(single[d]) else single[d])
        rows.append({"failure": failure, "held_out_seeds": auroc(weights, test, failure),
                     "unseen_failure": unseen, "best_single": best,
                     "best_single_auroc": single[best]})
    return pd.DataFrame(rows).set_index("failure")


def false_alarms(weights: dict, wide: pd.DataFrame, train: pd.DataFrame, rate: float = 0.05) -> pd.Series:
    """Share of each harmless control's windows above the threshold that gives `rate`
    false alarms on the training seeds' clean windows."""
    clean = train[train.failure.eq("clean")]
    line = float(np.quantile(predict(weights, clean), 1 - rate))
    controls = wide[~wide.expected_harmful & ~wide.failure.eq("clean")]
    return pd.Series(predict(weights, controls) > line, index=controls.index).groupby(
        [controls.failure, controls.severity]).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", required=True)
    parser.add_argument("--train-seeds", required=True, help="comma-separated; the other seeds are held out")
    parser.add_argument("--min-drop", type=float, default=0.0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--out", help="write weights trained on the train seeds here")
    args = parser.parse_args()

    df = pd.read_parquet(args.results)
    held_out = set(df.dataset) - {"nfcorpus"}
    if held_out:
        raise SystemExit(f"train only on NFCorpus results, these include {sorted(held_out)}")
    wide = window_table(df, min_drop=args.min_drop)
    seeds = [int(s) for s in args.train_seeds.split(",")]
    print(evaluate(wide, seeds, C=args.C).round(3).to_string())
    train = wide[wide.seed.isin(seeds)]
    weights = fit(train, C=args.C)
    test = wide[~wide.seed.isin(seeds)]
    print("\nfalse alarms on held-out seeds (threshold: 5% on training clean windows)")
    print(false_alarms(weights, test, train).round(2).to_string())
    if args.out:
        weights["trained_on"]["min_drop"] = args.min_drop
        Path(args.out).write_text(json.dumps(weights, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
