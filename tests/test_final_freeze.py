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
