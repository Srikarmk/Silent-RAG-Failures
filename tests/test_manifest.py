import hashlib
import json
import subprocess

import pytest

from silent_rag import manifest as manifest_module
from silent_rag.manifest import cpu_name, git_state, package_versions, write_manifest

ANALYSIS = {"question_bootstrap": {"resamples": 10, "seed": 0}, "window_bootstrap": {"resamples": 5, "seed": 1},
            "min_mean_ndcg10_drops_for_harmful_share": [0.0]}


def outputs_and_inputs(tmp_path):
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True, exist_ok=True)
    (out / "table.parquet").write_bytes(b"table bytes")
    (out / "sub" / "other.parquet").write_bytes(b"other")
    config = tmp_path / "run.yaml"
    config.write_text("model: m\n")
    return out, {"run_config": config}


def manifest(tmp_path, repo_dir=None):
    out, inputs = outputs_and_inputs(tmp_path)
    return out, write_manifest(out, run_id="abc", input_files=inputs, analysis_config=ANALYSIS,
                               status={"t1": "written", "t2": "skipped: no settings"}, seconds={"t1": 1.5},
                               repo_dir=repo_dir or tmp_path)


def test_hashes_match_hashlib_and_the_manifest_does_not_hash_itself(tmp_path):
    out, got = manifest(tmp_path)
    assert got["outputs"] == {"sub/other.parquet": hashlib.sha256(b"other").hexdigest(),
                              "table.parquet": hashlib.sha256(b"table bytes").hexdigest()}
    assert got["inputs"]["run_config"]["sha256"] == hashlib.sha256(b"model: m\n").hexdigest()
    _, again = manifest(tmp_path)  # a second write must not pick up manifest.json as an output
    assert again["outputs"] == got["outputs"]


def test_outside_a_git_checkout_commit_is_null_with_a_note(tmp_path):
    state = git_state(tmp_path)
    assert state["commit"] is None and state["uncommitted_changes"] is None and state["note"].startswith("not a git checkout")


def test_dirty_working_tree_is_reported(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
                       check=True, capture_output=True)
    git("init", "-q")
    (repo / "f.txt").write_text("one")
    git("add", "f.txt")
    git("commit", "-q", "-m", "first")
    clean = git_state(repo)
    assert len(clean["commit"]) == 40 and clean["uncommitted_changes"] is False and clean["note"] is None
    (repo / "f.txt").write_text("two")
    assert git_state(repo)["uncommitted_changes"] is True


def test_keys_sorted_status_timings_and_settings_recorded(tmp_path):
    out, got = manifest(tmp_path)
    text = (out / "manifest.json").read_text()
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"
    assert got["tables"] == {"t1": {"status": "written", "seconds": 1.5}, "t2": {"status": "skipped: no settings", "seconds": None}}
    assert got["bootstrap_settings"] == {k: ANALYSIS[k] for k in ("question_bootstrap", "window_bootstrap")}
    assert got["run_id"] == "abc" and set(got["machine"]) == {"platform", "cpu", "cpu_count", "omp_num_threads"}


def test_cpu_name_is_a_non_empty_string():
    assert isinstance(cpu_name(), str) and cpu_name()


@pytest.mark.parametrize("failure", [OSError("no sysctl"), subprocess.CalledProcessError(1, "sysctl")], ids=["missing", "fails"])
def test_a_failing_sysctl_falls_back_without_raising(monkeypatch, failure):
    def failing(*args, **kwargs):
        raise failure

    monkeypatch.setattr(manifest_module.sys, "platform", "darwin")  # take the sysctl path on any OS
    monkeypatch.setattr(manifest_module.subprocess, "run", failing)
    monkeypatch.setattr(manifest_module.platform, "processor", lambda: "fallback-cpu")
    assert cpu_name() == "fallback-cpu"


def test_package_versions_are_strings_or_null():
    versions = package_versions()
    assert set(versions) == {"python", "numpy", "pandas", "torch", "sentence-transformers", "faiss"}
    assert versions["numpy"] and all(v is None or isinstance(v, str) for v in versions.values())
