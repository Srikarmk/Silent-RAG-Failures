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
- `H_full_swap_minilm` alarmed on up to 45% (KS/PSI) and up to 95% (QPP) of windows.
- Mixed truncation stayed near the clean rate: permutation ≤ 14%.
- A permutation p-value says the scores changed, not that retrieval got worse. Neither rule
  can tell a harmless model change from a harmful one.

**F1 detection power, measured on harmful windows.** Every F1 window was harmful.

| | at severity 0.9 | at severity 0.5 |
|---|---|---|
| `ks_top1`, `ks_mean10`, `psi_top1`, `psi_mean10`, `qpp_max` | 100% under the permutation rule, both splits, both models; ≥ 88% under the other two rules | see below |
| `*_gap`, `*_std10`, `qpp_nqc` | weak: usually near their clean false-alarm rate | see below |

At severity 0.5 the percentile rule usually detected more:
- split A: `ks_top1` 40%/37% (BGE/MiniLM) against 25%/28% for the permutation rule
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
- Mixed truncation was harmful in 2.5–6.7% of windows, about the chance rate at α = 0.05,
  with a mean nDCG change of 0 or slightly better.
- Those windows are not false alarms. All three rules leave them out of the false-alarm
  denominator, as `detector_scoring.false_alarm_rate` does, and `compare_threshold_rules`
  reports them in `n_harmful_excluded`: per detector, 11 of the 180 mixed-truncation
  evaluation windows in split A (5/3/3 at severities 0.1/0.5/0.9) and 4 in split B (3/1/0).
  No clean, full-swap or topic-shift window was harmful, so nothing else is excluded.

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
- **No multiple-testing correction** across detectors.
- **One dataset and one window size.** F1 at only two severities.
