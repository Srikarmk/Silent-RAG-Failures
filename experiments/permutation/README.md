# Per-window permutation test: dev check (#73)

Dev-only. NFCorpus, no SciFact or FiQA, no full benchmark.

## What it compares

Each `FeatureDrift` detector (`ks_*`, `psi_*`, `qpp_*`, 11 in all) runs next to its `_perm` twin
on the same windows. The twin scores 1 − p, where p is the window's permutation p-value
(999 reshuffles, permutation seed 0).

| rule | alarm when | needs clean windows |
|---|---|---|
| permutation | p < 0.05 (`_perm` score > 0.95) | no |
| percentile | plain score > 95th percentile of clean scores on the calibration seeds | yes |
| clean max | plain score ≥ highest clean score on the calibration seeds (the site's rule) | yes |

The permutation and percentile rules are computed by `compare_threshold_rules`
(`silent_rag.eval.permutation`). The clean-max rule is not part of that function: it is
implemented in `analyze.py` (`clean_max_rule`) and scored on the same evaluation windows.

**Ground truth.** A window is harmful when its bootstrap p-value for the nDCG@10 drop is
below 0.05 and the drop is at least 0.02 (`min_drop = 0.02`). No detector sees this label.
The significance level of the ground truth (`harm_alpha`) is separate from the detectors'
alarm level (`alpha`): changing the alarm level never changes which windows are harmful.
The run stores `harmful` at `min_drop = 0`, so `analyze.py` re-thresholds it from the stored
`p_value` and `ndcg_drop` for every table. Nothing is rerun.

Runner seeds 0–5. **Split A** calibrates on seeds 0–2 and evaluates on 3–5, **split B** the
reverse. All rates are on evaluation seeds only, 60 windows per condition (3 seeds × 20 trials).

The conditions are:
- clean windows
- the negative controls `H_mixed_truncation_512` (0.1/0.5/0.9), `H_full_swap_minilm` and
  `H_full_swap_bge` (1.0), and `H_query_topic_shift` (0.1/0.5/0.9)
- F1 partial upgrade to MiniLM and to BGE at 0.5 and 0.9

## Reproduce

Needs the code from PR #76 (main ≥ 4d8f07e) plus the #73 detector changes. From the repository root:

```bash
python -m silent_rag.run --config experiments/permutation/dev_check.yaml --out results/permutation/dev_check.parquet
python experiments/permutation/analyze.py
python experiments/permutation/repeats.py
```

- `analyze.py` writes `summary.txt` and CSVs next to the parquet.
- `repeats.py` writes `repeats.txt` and `repeats.csv`. It needs the embedding cache from the run.
- Outputs go under `results/`, which is gitignored.
- The run is deterministic: two independent runs gave identical scores, nDCG drops and harm
  labels on all 10,800 rows they share.
- On a CPU laptop the run takes about 7.5 min with the embedding cache warm. Encoding the corpus
  takes several minutes more the first time.

## Findings

### Primary evidence: windows of distinct queries

These are the clean windows, mixed truncation, the full swaps and F1.

**The permutation p-values look calibrated or conservative on clean windows.** Across all
120 clean windows (6 seeds):
- p ≤ 0.05 occurred in 0.8–5.8% of windows, depending on the detector.
- On the evaluation seeds of either split, the permutation rule's clean false-alarm rate was
  at most 10% for every detector.
- This is a small sample with only 6 reference windows, so it shows no sign of the test
  rejecting too often. It does not show nominal calibration.

**The clean-calibrated thresholds were unstable with three calibration seeds.**
- Each seed has its own reference window, so a threshold from 60 clean windows over 3 reference
  windows can land far from where the other seeds need it.
- Percentile rule: clean false alarms ranged from 0% to 61.7% (`qpp_gap`, split B; 41.7% for
  `qpp_nqc`).
- Clean-max rule: up to 55% (`qpp_gap`) and 30% (`qpp_nqc`), both in split B.
- For KS and PSI the percentile rule stayed between 0% and 20%.
- This says the thresholds are unstable with this few calibration seeds. It does not say
  clean-window calibration fails in general.

**Harmless controls that change the model fire both rules.**
- `H_full_swap_bge` alarmed on 100% of windows for every detector except `qpp_max` (0%).
- `H_full_swap_minilm` alarmed on up to 45% (KS/PSI) and up to 85% (QPP) of windows under
  the permutation rule, and on up to 82% and 100% under the percentile rule (split B).
- Mixed truncation stayed near the clean rate: permutation ≤ 13.3%.
- A permutation p-value says the scores changed, not that retrieval got worse. Neither rule
  can tell a harmless model change from a harmful one.

**F1 detection power, measured on harmful windows.** 479 of the 480 F1 windows were harmful.
The exception is one MiniLM window at severity 0.5 (seed 5, nDCG@10 drop 0.019, below
`min_drop`). Seed 5 is an evaluation seed in split A, so that split's MiniLM 0.5 rates are over
59 windows.

| | at severity 0.9 | at severity 0.5 |
|---|---|---|
| `ks_top1`, `ks_mean10`, `psi_top1`, `psi_mean10`, `qpp_max` | 100% under the permutation rule, both splits, both models; ≥ 88% under the other two rules | see below |
| `*_gap`, `*_std10`, `qpp_nqc` | weak: usually near their clean false-alarm rate | see below |

At severity 0.5 the percentile rule usually detected more:
- split A: `ks_top1` 40%/37% (BGE/MiniLM) against 25%/29% for the permutation rule
- split B: `ks_top1` 77%/73% against 62%/52%
- Part of that gap comes with a higher clean false-alarm rate (split A `ks_top1`: 11.7%
  against 1.7%). But it persists where both rules had 0% clean false alarms (split B
  `ks_top1`, `ks_mean10`).
- So the conservative permutation rule gives up some power at moderate severity.
- Read power next to each rule's clean false-alarm rate: where a rule's clean rate was high,
  its F1 rate was too (for example `qpp_gap` in split B) and says little about detection.
  `summary.txt` puts the two side by side.

**The ground truth confirms the controls are harmless.**
- No full-swap or topic-shift window was harmful, and the full swaps improved nDCG@10
  (mean drop −0.051 for BGE, −0.025 for MiniLM).
- No mixed-truncation window was harmful either. Its mean nDCG@10 change was 0 or slightly
  better (mean drop −0.000/−0.002/−0.004 at severities 0.1/0.5/0.9).
- With `min_drop = 0` (bootstrap significance alone), 2.5–6.7% of mixed-truncation windows
  were labeled harmful, about the chance rate at 0.05. Every one of those drops was below
  0.014, so `min_drop = 0.02` labels them harmless.
- So no clean or control window is harmful, and `n_harmful_excluded` is 0 everywhere. A
  control window the ground truth marks harmful would be left out of the false-alarm
  denominator, as `detector_scoring.false_alarm_rate` does, by all three rules.

**Cost (#27).** The runner records `cost_s` for the `_perm` detectors like any other detector.

| detector family | median per window at 999 permutations | vs plain detector |
|---|---|---|
| KS | ≈ 20 ms | ≈ 15–22× |
| PSI | ≈ 41 ms | ≈ 105–130× (refits its bins on every reshuffle) |
| QPP | ≈ 6 ms | ≈ 20× |

Cost scales linearly with `permutations`. All 11 `_perm` detectors together cost ≈ 0.26 s per
window, 413 s of this run's 424 s of detector scoring.

### Limitation: repeated queries (`H_query_topic_shift`)

**This control's windows are not distinct queries, so it is reported separately.**
- It oversamples one topic and repeats some of that topic's queries. At severity 0.9 a window
  of 100 asks about 54 distinct queries.
- The permutation test assumes the reference and window values are exchangeable,
  independent draws. Repeated queries break that.
- It is not used as calibration evidence, and the code does not de-duplicate anything.

**The topic shift fires both rules on many windows, and the rate depends heavily on the seed.**
- KS/PSI permutation rule at severity 0.9: 35–85% (split A) and 7–75% (split B).
- Per seed, from about 0% to 100%.

**`repeats.py` sizes the effect of the repeats themselves.** This is a diagnostic, not a method.
It rescores the same windows three ways:
- as the runner did (reproduced exactly)
- with one row per distinct query
- with the same repeat rate drawn from a random set of pool queries that is not a topic

With repeats but no topic shift, at severity 0.9 the permutation rule still alarmed on:
- KS: 5–28% of windows
- PSI: 15–55%
- QPP: 0–22%

The percentile rule was also inflated (for example `psi_gap` 45–57%). At severity 0.1 both
rules stayed near their own clean rates.

**So neither rule is calibrated for repeated traffic, PSI least of all.** Results on windows
with repeated queries can't be read as false-alarm rates under the test's null. Handling them
(a test that respects the repeats, or de-duplicating by design) is out of scope for #73.

### Other limitations

- **Small sample.** 6 reference windows, and windows within a seed overlap heavily (100 of a
  223-query pool), so the per-seed rates are noisy.
- **Resolution.** The p-value has a resolution of 0.001, and every window uses the same
  reshuffles.
- **KS p-values can be conservative.** The KS statistic takes only a few discrete values, so
  many reshuffles reproduce the observed statistic exactly. Those ties count toward the tail:
  p = (1 + reshuffles ≥ observed) / (permutations + 1). That makes p ≥ the probability of a
  strictly larger statistic, so the KS permutation rule can alarm less often than its nominal
  α and lose some power. This is consistent with the low KS clean rates above (`ks_top1` and
  `ks_mean10` p ≤ 0.05 in 0.8% of clean windows). A mid-p or randomized tie-break would
  reduce this; neither is implemented here.
- **No multiple-testing correction** across detectors.
- **One dataset and one window size.** F1 at only two severities.
