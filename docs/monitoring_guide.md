# Monitoring retrieval quality without labels

A practical guide for teams running RAG: what to watch, which check catches which failure, where to set the alarm, and what each check costs. Short on purpose.

> **Draft (#32).** Each section has an owner, following the split in #32. Numbers marked *dev* are from NFCorpus (or MLDR for F3) and get replaced by held-out numbers from the final run (#29) before this is done. Sections still to write are marked **TODO**.

## In one paragraph

**TODO (team, after #29).** The recommendation, in three or four sentences.

## The failures, and the changes that aren't failures

**TODO (Shane).** F1 to F5 in a sentence each: how they happen in practice and how much damage they do. Then the negative controls: big changes that don't hurt retrieval, and why a monitor that fires on them is a problem.

## Which check catches which failure

**TODO (team).** One table, detector × failure, from the #30 heatmap and #29.

## Score detectors, QPP, canaries, LLM judge, and the PSI rule

**TODO (Alekhya).** Including what PSI > 0.25 means in practice (#31).

## Embedding detectors, ground truth, baseline

**TODO (Jenna).** Including how "harmful" is defined (`min_drop`, #52).

## The re-embedding check

*Owner: Srikar.*

`re_embed` re-encodes a sample of the index's documents with the model the system is supposed to be running, and measures how far the stored vectors are from the fresh ones.

- **It catches a mixed embedding model almost perfectly.** On F1 (part of the index embedded with a different model) its AUROC is 1.00 against clean windows, at every partner model (*dev*, #79).
- **It fires on harmless model changes too.** A full swap to another model, or re-embedding at a longer `max_seq_length`, changes every vector without hurting retrieval. re_embed alarms on 100% of those windows (*dev*, #79). It detects *change*, not *damage*.
- **It is blind to failures where every vector is right for its text.** On F3 (part of a chunked index falling back to one vector per document) it scores AUROC 0.50: every row is the correct embedding of its own text, so there's nothing to find. That's true even though F3 costs up to 0.32 nDCG@10 (*dev, MLDR*, #82). Distractor documents (F2) are the same: AUROC 0.50.
- **It's the most expensive cheap check:** about 3 s per check for a 200-document sample, about 3,000× a KS test (*dev*, #27).

**When to use it:** as the check to run after an embedding-model change, to confirm every document was re-embedded. Don't use it as a quality alarm. A monitor built on it alone pages someone after every planned model upgrade, and misses failures that aren't about embeddings.

## Combining checks and two-stage monitoring

*Owner: Srikar.*

**Combining the cheap checks** (#75, #80). `combined_scores` is a logistic regression over the 11 score-based detectors (KS, PSI and QPP on the retrieval scores), trained on dev windows that include the negative controls, so it learns damage rather than change.

- It beats the best single cheap detector by about 0.04 AUROC on F1 (0.90 against 0.86 for `qpp_max`, *dev*), and costs 6 ms a check.
- It still fires on harmless changes: 13–26% false alarms on clean and control windows, depending on the threshold. Like every score detector, it misses F2.

**Two-stage monitoring** (#74). Run the cheap check on every window. Only when it fires, ask an LLM judge whether the retrieved documents are actually relevant. A drift alarm says something changed; the judge says whether retrieval got worse.

- **The rule we use.** Both thresholds are set on clean windows only, no labels:
  - stage 1 (`combined_scores`) at 20% clean false alarms; looser than it would be alone, because stage 2 removes most of its false alarms
  - the judge at 10% clean false alarms (0.742 on NFCorpus)
- **What it buys** (*dev*, test seeds 7–9):

  | Monitor | Catches harmful windows | False alarms |
  |---|---|---|
  | `combined_scores` alone | 39% | 13.3% |
  | `combined_scores` → judge | **54%** | **5.2%** |

  No false alarms on clean windows. The ones that remain are all on query-topic shifts: an absolute-quality judge reads harder questions as worse retrieval.
- **The judge needs a big enough sample.** With 10 queries × top 5 documents per window it was too noisy (AUROC 0.74 on F1). With 100 × 5 it separates F1 from clean at 0.90–0.99 (*dev*).
- **Cost.** The judge runs on about a third of windows, so the pair costs about a third of judging every window. On our machine with a free local model (qwen3:30b-a3b) that's about 107 s per window if nothing is cached (*dev*, #27). In practice the same question-document pairs repeat, so a cache cuts that a lot.
- **What it doesn't fix.** Small damage (F1 at 10% of the index) and distractor documents (F2) stay mostly undetected. The judge averages over too many pairs to see a few displaced documents.

## What each check costs

*Owner: Srikar.*

Median time per check on one window of 100 queries. Apple M5 Pro, CPU, one process, nothing else running (*dev*, #27). Everything here costs $0; the judge runs on a local model.

| Check | Median per window |
|---|---|
| QPP (`qpp_max`, `qpp_nqc`, `qpp_gap`), PSI | 0.25–0.30 ms |
| KS | 0.7–1.1 ms |
| `combined_scores` | 6 ms |
| Domain classifier, queries / retrieved documents | 21 ms / 105 ms |
| `re_embed` (200 documents) | 3 s |
| LLM judge, 100 queries × top 5 | ~315 s uncached (0.63 s per pair) |
| Two-stage, `combined_scores` → judge | ~107 s uncached |

Times vary with machine load: a timing run that shared the machine had p90s up to 2× the median, against within 10% here. Time on an otherwise idle machine.

## Running the benchmark

*Owner: Srikar.*

```bash
pip install -e ".[dev]"
python -m silent_rag.run --config configs/smoke.yaml        # end-to-end check, a few minutes
python -m silent_rag.run --config configs/dev.yaml          # NFCorpus
python -m silent_rag.eval.report results/dev.parquet        # tables as Markdown
```

- Adding a failure or a detector: see the README.
- To reproduce our numbers exactly, install `requirements-lock.txt` first. Newer library versions gave identical nDCG drops and p-values in our check (#34, #83).
- The held-out run only starts from a clean, tagged commit (`--final`, #81), and records the commit, machine and package versions next to its results.

## What we didn't test

**TODO (team).** Including the agentic-retriever signal (#35) if it makes it in.
