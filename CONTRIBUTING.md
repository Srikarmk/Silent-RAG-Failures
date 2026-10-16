# How we're working on this

Keeping this short so we actually follow it. If something here gets in the way, bring it up and we'll change it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

`pre-commit install` sets up nbstripout, which clears notebook outputs when you commit. Without it, two people touching the same notebook means a merge conflict every time.

On a Mac, if anything that uses both faiss and torch just exits with no error, run it with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`.

## Who's on what (for now)

| Area | Person | Code |
|---|---|---|
| Data + baseline | Jenna | `data`, `eval/ground_truth.py` |
| Pipeline + runner | Srikar | `pipeline.py`, `run.py`, `eval/results.py` |
| Failure injection | Shane | `inject/` |
| Detectors | Alekhya | `detect/` |

This isn't fixed. Once injection is stable (around week 4), Shane moves over to help with detectors since that's the biggest chunk. Weeks 8 to 10 we're all on the held-out run and the guide.

## How the code fits together

The whole thing hangs on three interfaces in `src/silent_rag/`:

- **`Injector.apply(state, severity, seed)`** returns a damaged copy of the retrieval state. Severity always means fraction affected (0.1 to 0.9). Same seed, same result.
- **`Detector.fit(reference)` / `Detector.score(window)`** take an `Observation`, which has query embeddings, top-k ids and scores, but **no qrels**. That's on purpose. If a detector needs labels, it's not label-free and it doesn't belong in this project.
- **`ResultRow`** is one row per condition, trial and detector. Every plot comes from this table, so if you need a new column, add it there rather than making a side CSV.

Qrels only get touched in `eval/ground_truth.py`.

## The held-out rule

We tune only on NFCorpus. SciFact and FiQA are for the final numbers. The runner won't touch them unless you pass `--final`. Please don't work around it, even "just to check something". The moment we tune on them, the results stop meaning anything.

## Branches and PRs

- Don't push straight to `main`. Branch, open a PR, get one review, squash merge.
- Name branches `<area>/<what>`, like `inject/f1-partial-upgrade` or `detect/mmd`.
- Keep branches small and short. A few days, not a few weeks.
- Each PR should close an issue (`Closes #12` in the description).
- Notebooks are for exploring. Anything we'll reuse goes in `src/` and gets imported into the notebook.
- Don't commit data, embeddings or caches. They're all in `.gitignore` and can be rebuilt. Small result summaries are fine.

## Issues

- One issue should be about 1 to 3 days of work. If it's bigger, split it.
- Fill in "Done when" so we know when to close it.
- Labels: `area/*` for which part of the code, `research` for decisions we need to make together (like window size).
- Milestones match the timeline in the README.

We'll do a quick weekly sync where we go through the board and see what's stuck.
