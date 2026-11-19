# Held-out evaluation runbook (SciFact, FiQA)

Run by the person responsible for the final evaluation, on the agreed date. Nothing here has been
run on held-out data.

`configs/final.yaml` lists both datasets; the analysis takes one dataset per run, so it is run as
two configs with identical settings:

| Config | Dataset | Expected run_id |
|---|---|---|
| `experiments/final/final_scifact.yaml` | SciFact (5,183 docs, 300 test questions, no validation split) | `bebcd2b7cf3f` |
| `experiments/final/final_fiqa.yaml` | FiQA (57,638 docs, 648 test, 500 validation) | `9bd99cb1bd89` |

Before running: settle the open decisions (seeds; SciFact's reference split). If the configs change,
re-check the run_ids with `run_id_from_run_config`.

## 1. Runner (one run per dataset)

    python -m silent_rag.run --config experiments/final/final_scifact.yaml --final --out results/final_scifact.parquet
    python -m silent_rag.run --config experiments/final/final_fiqa.yaml    --final --out results/final_fiqa.parquet

(Check the output flag with `python -m silent_rag.run --help`.)

## 2. Analysis (one per dataset; clean committed tree)

    HF_HUB_OFFLINE=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -m silent_rag.analyze_benchmark_results \
      --config experiments/final/final_scifact.yaml --analysis experiments/final/final.analysis.yaml \
      --results results/final_scifact.parquet --out results/final_scifact/ --final \
      --skip statistic_all_test_questions_vs_validation_reference_by_severity

    HF_HUB_OFFLINE=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -m silent_rag.analyze_benchmark_results \
      --config experiments/final/final_fiqa.yaml --analysis experiments/final/final.analysis.yaml \
      --results results/final_fiqa.parquet --out results/final_fiqa/ --final

SciFact skips table 10: it has no validation split to fit the statistics on.

## 3. Record

For each dataset: check `manifest.json` (version 2, commit_changed_during_run false, no
uncommitted paths), copy it to `experiments/analysis_f1/manifests/`, add a README row, commit.

## Timing note

No SciFact or FiQA embeddings are cached (checked 2026-11-19): the first run embeds both corpora with
the baseline and both replacement models (3 x 62,821 documents; FiQA is most of it). Expect embedding to
dominate the runner's time; later runs on the same configs reuse the cache.
