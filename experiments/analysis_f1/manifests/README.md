# Analysis manifests

Each file is a copy of `manifest.json` from one run of
`silent_rag.analyze_benchmark_results` on `experiments/analysis_f1/analysis_f1.yaml`
(run_id 16f99a9a85d4).

| File | Output folder | Analysis code | Notes |
|---|---|---|---|
| `analysis_f1_2026-11-15.json` | `results/analysis_f1/` | `5185830` | Clean tree at start and end. |
| `analysis_f1_v2_2026-11-16.json` | `results/analysis_f1_v2/` | `f1db7f0` (step 18c-ii) | The manifest records `0168d46` with `uncommitted_changes: true`: the git state is read when the manifest is written, at the end of the run, and plotting commits (19a-i to 19b-ii-a) and untracked plotting files were added during the run. No commit after `f1db7f0` changes `src/silent_rag` outside `plots/` or either analysis config, and the analysis does not import `plots/`. |

Manifests without a `manifest_version` key are version 1: `git.commit` and `git.uncommitted_changes` were read when the manifest was written, at the end of the run. Both manifests above are version 1. Version 2 (`manifest_version: 2`) records the checkout at the start of the run (`git.commit`, `git.uncommitted_changes`, `git.uncommitted_paths`), the checkout when the manifest is written (`git.end_commit`, `git.uncommitted_paths_at_end`), and `git.commit_changed_during_run`, true when the two commits differ.
