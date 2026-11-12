#!/usr/bin/env python
"""Run the configuration written by notebook 08 §1c as a standalone process, kept awake with caffeinate.

Lock the screen instead of logging out; keep the Mac on AC power with the lid open.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def run_id_of(cfg_path: Path) -> str:
    from silent_rag import run as R  # the same hash run.run writes in every row (notebook 08 §1c)
    hashed = {**R.DEFAULTS, **yaml.safe_load(cfg_path.read_text())}
    hashed.pop("seed", None)
    return R.run_id(hashed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runner-log", type=Path, required=True)
    p.add_argument("--interval", type=int, default=60, help="seconds between load samples")
    a = p.parse_args()

    run_id = run_id_of(a.config)
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}
    with open(a.runner_log, "w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "silent_rag.run", "--config", str(a.config), "--out", str(a.out)],
                                cwd=a.root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    # caffeinate holds off sleep until the runner exits; the runner PID stays the one §8 excludes
    subprocess.Popen(["caffeinate", "-ims", "-w", str(proc.pid)], start_new_session=True)
    print(f"runner_pid {proc.pid}\nrun_id {run_id}\nstarted {now()}", flush=True)

    while proc.poll() is None:
        top = subprocess.run(["ps", "-Ao", "pcpu=,pid=,comm=", "-r"],
                             capture_output=True, text=True).stdout.splitlines()[:8]
        print(f"sample {now()} loadavg {os.getloadavg()}", *top, sep="\n", flush=True)
        time.sleep(a.interval)

    print(f"exit_status {proc.returncode}\nfinished {now()}", flush=True)
    if proc.returncode == 0:
        found = sorted(set(pd.read_parquet(a.out).run_id))
        print("run_id check:", "ok" if found == [run_id] else f"MISMATCH {found} vs {run_id}", flush=True)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()