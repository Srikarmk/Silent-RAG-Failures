"""Provenance manifest for an analysis output directory (spec 5.8): what produced these tables.

Keys are sorted, so two runs of the same command give the same manifest except for the timing
fields (and the git state, if the code changed in between).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

# package -> distribution names to try (faiss ships as faiss-cpu or faiss-gpu)
PACKAGES = {"numpy": ["numpy"], "pandas": ["pandas"], "torch": ["torch"],
            "sentence-transformers": ["sentence-transformers"], "faiss": ["faiss-cpu", "faiss-gpu", "faiss"]}
BOOTSTRAP_SETTINGS = ["question_bootstrap", "window_bootstrap", "seed_cluster_bootstrap", "question_vulnerability"]
MANIFEST = "manifest.json"


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo_dir) -> dict:
    """Commit of repo_dir's checkout and whether its working tree has uncommitted changes;
    both null with a note when repo_dir is not in a git checkout (or git is missing)."""
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=True).stdout
        commit = git("rev-parse", "HEAD").strip()
        return {"commit": commit, "uncommitted_changes": bool(git("status", "--porcelain").strip()), "note": None}
    except (OSError, subprocess.CalledProcessError) as e:
        reason = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
        return {"commit": None, "uncommitted_changes": None, "note": f"not a git checkout: {reason}"}


def package_versions() -> dict:
    """Installed versions, read from package metadata without importing; null when not installed."""
    versions = {"python": platform.python_version()}
    for package, distributions in PACKAGES.items():
        versions[package] = None
        for distribution in distributions:
            try:
                versions[package] = metadata.version(distribution)
                break
            except metadata.PackageNotFoundError:
                continue
    return versions


def cpu_name() -> str:
    """The CPU's name (notebook 08's cpu_name): on macOS from sysctl machdep.cpu.brand_string (e.g.
    "Apple M3 Pro"), elsewhere or on any failure platform.processor(), then platform.machine()."""
    if sys.platform == "darwin":
        try:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  check=True, timeout=5).stdout.strip()
            if name:
                return name
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "unknown"


def machine() -> dict:
    return {"platform": platform.platform(), "cpu": cpu_name(),
            "cpu_count": os.cpu_count(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}


def write_manifest(out_dir, *, run_id: str, input_files: dict, analysis_config: dict, status: dict, seconds: dict,
                   repo_dir=".") -> dict:
    """Write out_dir/manifest.json and return its content.

    input_files: {role: path} for the run config, analysis config and results parquet; every file
    already in out_dir (other than the manifest itself) is hashed too. status and seconds:
    {table: "written" | "skipped: <reason>"} and {table: wall-clock seconds}, from the caller.
    """
    out_dir = Path(out_dir)
    outputs = {p.relative_to(out_dir).as_posix(): sha256(p)
               for p in sorted(out_dir.rglob("*")) if p.is_file() and p.name != MANIFEST}
    content = {
        "git": git_state(repo_dir),
        "run_id": run_id,
        "inputs": {role: {"path": str(path), "sha256": sha256(path)} for role, path in input_files.items()},
        "outputs": outputs,
        "packages": package_versions(),
        "machine": machine(),
        "bootstrap_settings": {k: analysis_config[k] for k in BOOTSTRAP_SETTINGS if k in analysis_config},
        "tables": {name: {"status": outcome, "seconds": seconds.get(name)} for name, outcome in status.items()},
        "python_executable": sys.executable,
    }
    (out_dir / MANIFEST).write_text(json.dumps(content, sort_keys=True, indent=2) + "\n")
    return content
