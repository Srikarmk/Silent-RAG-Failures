"""Experiment runner.

    python -m silent_rag.run --config configs/dev.yaml
    python -m silent_rag.run --config configs/final.yaml --final
"""

from __future__ import annotations

import argparse

import yaml

DEV_DATASETS = {"nfcorpus"}
HELD_OUT = {"scifact", "fiqa"}


def check_datasets(datasets: list[str], final: bool) -> None:
    """We only tune on NFCorpus. Touching the held-out sets needs --final."""
    held_out = HELD_OUT.intersection(datasets)
    if held_out and not final:
        raise SystemExit(
            f"{sorted(held_out)} are held out. Tune on NFCorpus, and pass --final "
            "only for the real evaluation run."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    check_datasets(cfg["datasets"], args.final)

    # TODO (#7): for each dataset x failure x severity x trial:
    #   build state -> inject -> search -> Observation + per-query nDCG
    #   -> run every detector -> append ResultRow -> write parquet
    raise NotImplementedError


if __name__ == "__main__":
    main()
