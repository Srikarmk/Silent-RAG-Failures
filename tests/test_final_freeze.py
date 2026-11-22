"""--final runs only from frozen, tagged code (#29)."""

import functools
import subprocess

import pytest

from silent_rag import run as runner


def sh(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "configs").mkdir()
    (r / "src" / "a.py").write_text("x = 1\n")
    (r / "configs" / "final.yaml").write_text("datasets: [scifact]\n")
    sh(r, "init", "-q")
    sh(r, "add", ".")
    sh(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "frozen")
    sh(r, "tag", "final-v1")
    return r


def check(repo, out=None, config="configs/final.yaml"):
    return runner.check_frozen(str(repo / config), out or repo / "results" / "final.parquet",
                               git=functools.partial(runner.git, repo=repo), root=repo)


def test_a_clean_tagged_checkout_passes_and_is_recorded(repo):
    got = check(repo)
    assert got["tag"] == "final-v1" and len(got["commit"]) == 40
    assert got["config"] == "configs/final.yaml" and len(got["config_sha1"]) == 40


def test_an_edited_file_refuses(repo):
    (repo / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="differ from HEAD"):
        check(repo)


def test_an_untracked_detector_refuses(repo):
    (repo / "src" / "new_detector.py").write_text("")
    with pytest.raises(SystemExit, match="new_detector.py"):
        check(repo)


def test_changes_outside_src_and_configs_are_fine(repo):
    (repo / "notes.md").write_text("draft")
    check(repo)


def test_an_untagged_head_refuses(repo):
    sh(repo, "tag", "-d", "final-v1")
    sh(repo, "tag", "v1")  # not a final tag
    with pytest.raises(SystemExit, match="final-v<N>"):
        check(repo)


def test_a_tag_on_an_older_commit_refuses(repo):
    (repo / "src" / "a.py").write_text("x = 3\n")
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "after the tag")
    with pytest.raises(SystemExit, match="final-v<N>"):
        check(repo)


def test_a_config_outside_configs_refuses(repo, tmp_path):
    other = tmp_path / "elsewhere.yaml"
    other.write_text("datasets: [scifact]\n")
    with pytest.raises(SystemExit, match="config under"):
        runner.check_frozen(str(other), repo / "out.parquet", git=functools.partial(runner.git, repo=repo), root=repo)


@pytest.mark.parametrize("existing", ["final.parquet", "final.json"])
def test_an_existing_result_refuses(repo, existing):
    (repo / "results").mkdir()
    (repo / "results" / existing).write_text("")
    with pytest.raises(SystemExit, match="happens once"):
        check(repo)


def test_not_a_git_checkout_refuses(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "final.yaml").write_text("")
    with pytest.raises(SystemExit, match="git checkout"):
        check(tmp_path)


def test_the_cli_checks_before_running(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(runner, "check_frozen", lambda *a: (_ for _ in ()).throw(SystemExit("frozen check")))
    monkeypatch.setattr(runner, "run", lambda *a, **k: called.append(1))
    monkeypatch.setattr("sys.argv", ["run", "--config", str(runner.ROOT / "configs" / "final.yaml"), "--final",
                                     "--out", str(tmp_path / "x.parquet")])
    with pytest.raises(SystemExit, match="frozen check"):
        runner.main()
    assert not called


# --final in parts: each --only-seeds part writes a sidecar, --merge checks and combines them

FROZEN_CODE = {"commit": "a" * 40, "tag": "final-v1", "config": "configs/final.yaml", "config_sha1": "b" * 40}


def fake_run(cfg, final=False, only_seeds=None):
    import pandas as pd

    return pd.DataFrame([dict(run_id="r1", seed=s, failure="clean", severity=0.0, trial=t, detector="d", score=0.1)
                         for s in only_seeds for t in range(2)])


def part(monkeypatch, tmp_path, seeds, code=FROZEN_CODE):
    out = tmp_path / f"final_p{seeds[0]}.parquet"
    monkeypatch.setattr(runner, "check_datasets", lambda *a: None)
    monkeypatch.setattr(runner, "check_frozen", lambda *a: dict(code))
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["run", "--config", str(runner.ROOT / "configs" / "final.yaml"), "--final",
                                     "--only-seeds", ",".join(map(str, seeds)), "--out", str(out)])
    runner.main()
    return out


def merge(monkeypatch, out, *parts, final=False):
    monkeypatch.setattr("sys.argv", ["run", "--merge", *map(str, parts), "--out", str(out)] + (["--final"] if final else []))
    runner.main()


def test_final_parts_merge_with_a_sidecar(monkeypatch, tmp_path):
    import json

    import pandas as pd

    a, b = part(monkeypatch, tmp_path, [0, 1]), part(monkeypatch, tmp_path, [2])
    assert json.loads(a.with_suffix(".json").read_text())["seeds"] == [0, 1]
    out = tmp_path / "final.parquet"
    merge(monkeypatch, out, a, b, final=True)
    assert sorted(pd.read_parquet(out).seed.unique()) == [0, 1, 2]
    side = json.loads(out.with_suffix(".json").read_text())
    assert {k: side[k] for k in FROZEN_CODE} == FROZEN_CODE
    assert side["rows"] == 6 and side["run_ids"] == ["r1"]
    assert [p["seeds"] for p in side["parts"]] == [[0, 1], [2]]


@pytest.mark.parametrize("key", ["commit", "tag", "config_sha1"])
def test_parts_from_different_code_refuse(monkeypatch, tmp_path, key):
    a = part(monkeypatch, tmp_path, [0])
    b = part(monkeypatch, tmp_path, [1], code={**FROZEN_CODE, key: "other"})
    with pytest.raises(SystemExit, match=f"different code: {key}"):
        merge(monkeypatch, tmp_path / "final.parquet", a, b)
    assert not (tmp_path / "final.parquet").exists()


def test_a_part_without_a_sidecar_refuses(monkeypatch, tmp_path):
    a, b = part(monkeypatch, tmp_path, [0]), part(monkeypatch, tmp_path, [1])
    b.with_suffix(".json").unlink()
    with pytest.raises(SystemExit, match="no sidecar"):
        merge(monkeypatch, tmp_path / "final.parquet", a, b)


def test_a_final_merge_does_not_overwrite(monkeypatch, tmp_path):
    a, b = part(monkeypatch, tmp_path, [0]), part(monkeypatch, tmp_path, [1])
    out = tmp_path / "final.parquet"
    merge(monkeypatch, out, a, b)
    with pytest.raises(SystemExit, match="happens once"):
        merge(monkeypatch, out, a, b)


def test_final_merge_needs_final_parts(monkeypatch, tmp_path):
    import pandas as pd

    p = tmp_path / "dev_p0.parquet"
    fake_run({}, only_seeds=[0]).to_parquet(p)
    with pytest.raises(SystemExit):
        merge(monkeypatch, tmp_path / "out.parquet", p, final=True)


def test_a_dev_merge_writes_no_sidecar(monkeypatch, tmp_path):
    p0, p1 = tmp_path / "dev_p0.parquet", tmp_path / "dev_p1.parquet"
    fake_run({}, only_seeds=[0]).to_parquet(p0)
    fake_run({}, only_seeds=[1]).to_parquet(p1)
    out = tmp_path / "dev.parquet"
    merge(monkeypatch, out, p0, p1)
    assert out.exists() and not out.with_suffix(".json").exists()
