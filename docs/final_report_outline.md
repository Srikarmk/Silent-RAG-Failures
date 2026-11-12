# Final report: outline (#33)

Section by section: what goes in, where its numbers come from, and what's still missing. A planning document, not the report. Status as of 2026-11-12.

**Rules from #33**
- Results come from the held-out run (#29) only. Dev numbers (NFCorpus, MLDR) go in Methods, as the evidence behind design choices, never in Results.
- Methods describe what we did, not what the proposal planned.
- Everyone reads the full report before we submit.

**Status key:** ✅ done and merged · 🟡 done, in an open PR · 🔴 not done yet · ⏳ waits on the held-out run

## 1. Introduction

- How RAG works, and why retrieval can get worse without any error: same story as the progress-update decks.
- The question: do label-free signals catch *damage*, or only *change*?
- Contributions, one line each:
  1. a benchmark that injects realistic retrieval failures at controlled severity, with negative controls
  2. the detector × failure results on held-out data
  3. a practical monitoring guide (#32, #84)

Status: ✅ text exists in the README and decks. Contribution 2 ⏳.

## 2. Related work

- The BEIR datasets, MLDR, and HoH (README references).
- Drift detection in ML: PSI, KS, MMD, domain classifiers.
- Query performance prediction in IR: score-based signals.
- LLM-as-judge for relevance.

Status: 🔴 not written. Owner: whoever wrote each detector.

## 3. Benchmark design (Methods)

### 3.1 Pipeline and datasets
- multi-qa-MiniLM-L6, pinned revision, one vector per doc truncated at 256 tokens (#5); FAISS exact search.
- Dev set: NFCorpus. Held out: SciFact and FiQA, touched once (#29, #81). MLDR for the chunking failure only (#12): its 20k-doc subsample, and why.

Status: ✅ (MLDR 🟡 #82)

### 3.2 Ground truth
- nDCG@10 drop, one-sided paired bootstrap per window, harmful = p < 0.05 and drop ≥ `min_drop`.
- `min_drop` 0.02, with 0.01 as a sensitivity row for F2 (#52). Evidence: no negative control comes out harmful at ≥ 0.02.
- Window of 100 queries (#4), 20 trials, 10 seeds.

Status: 🟡 `min_drop` still needs Jenna's agreement on #52.

### 3.3 Failures

| ID | Failure | Status |
|---|---|---|
| F1 | partial embedding upgrade, four partner models | ✅ / 🟡 #79 |
| F2 | distractor documents | 🟡 #76 |
| F2 | stale documents (HoH half) | 🔴 #11, Shane |
| F3 | chunked index partly falling back to one vector, MLDR only | 🟡 #82 |
| F4 | query shift outside the corpus | 🔴 #13, waiting on the reference-standard question |
| F5 | config mismatches | 🔴 #14, Jenna |

What each costs in nDCG (dev) goes here as design evidence.

### 3.4 Negative controls
- mixed truncation 256→512
- full model swap to MiniLM and to bge
- in-domain topic shift

Why they matter: AUROC pools them as negatives, so a detector that fires on harmless change is penalized. Status: 🟡 #76.

### 3.5 Detectors

| Family | Detectors | Status |
|---|---|---|
| Score drift | KS and PSI on 4 score features | ✅ |
| QPP | `qpp_max`, `qpp_nqc`, `qpp_gap` | ✅ |
| Embedding | domain classifier on queries and on retrieved docs | ✅ |
| Embedding | MMD | 🔴 #20, Shane |
| Re-embedding check | `re_embed` | ✅ |
| Canary queries | | 🔴 #24, Alekhya |
| LLM judge | local model (qwen3:30b-a3b), 100 queries × top 5 | 🟡 #77 |
| Combined | `combined_scores`, weights trained on NFCorpus | 🟡 #80 |
| Two-stage | `combined_scores` → judge, with the frozen rule | 🔴 detector written, not yet in a PR |

### 3.6 Metrics
- AUROC with seed-block CIs; negatives are clean windows plus harmless controls.
- False-alarm rate on each control, at thresholds set on clean windows (#23).
- Earliest reliably detected severity.
- Cost per check (#27).

How thresholds carry over to the held-out sets, frozen from NFCorpus or recalibrated on each dataset's clean windows, is still open on #29.

## 4. Results (held-out only)

All ⏳ #29. Planned tables and figures:

1. **Detector × failure AUROC heatmap** with CIs (#30, Alekhya), per held-out dataset.
2. **False alarms on the negative controls:** the change-vs-damage table.
3. **Earliness:** lowest severity each detector reliably catches.
4. **Cost vs detection:** median cost per check against mean AUROC, log x-axis.
5. **Two-stage monitor:** detection and false alarms for `combined_scores` alone and paired with the judge, under the frozen rule.
6. **F3 on MLDR:** clearly labelled as a dev result, since there's no held-out chunked set.

`python -m silent_rag.eval.report results/final.parquet` (#83) produces 1–3 and the cost table directly.

## 5. Findings we expect to test on held-out data

From dev results. The report says whether each one held up, whichever way it goes:
- **re_embed detects change, not damage.** AUROC 1.00 on F1, 100% false alarms on harmless model changes, blind to F2 and F3.
- **No cheap detector catches F2,** alone or combined.
- **On F3 the free QPP signal works** (0.92 AUROC at severity 0.3), while re_embed is blind.
- **The two-stage monitor beats the cheap combination on both counts:** 54% vs 39% detection, 5.2% vs 13.3% false alarms.
- **Small damage** (F1 at 10% of the index) goes undetected by everything.

## 6. Limitations

- **One retriever model and one chunk size.** Everything uses multi-qa-MiniLM-L6.
- **F3 is dev-only,** on a 20k-doc MLDR subsample.
- **The LLM judge is a 30B local model,** with kappa 0.54 against the qrels on NFCorpus (#26). A stronger judge might do better, and wasn't tested, to keep it free.
- **Thresholds come from small samples.** For example, the judge's rests on 140 clean windows.
- **Windows draw from a fixed pool of queries,** so windows overlap and CIs are over seeds, not over query populations.
- **What we didn't get to:** F4 and F5 if they don't land before the freeze, MMD, canaries, and the agentic-retriever signal (#35).

## 7. Conclusion and the guide

A short version of the monitoring guide's recommendation (#32), pointing to the full guide.

## Before we write: what has to land

1. Open PRs merged: #76–#84.
2. `min_drop` agreed (#52).
3. Thresholds rule decided (#29).
4. F2 stale documents, F4 and F5 in or explicitly cut (#11, #13, #14).
5. Freeze, tag `final-v1`, run `--final` once (#29, #81).
6. Heatmap and figures (#30).
