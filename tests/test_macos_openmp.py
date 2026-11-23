import importlib
import sys

import silent_rag


def reload_on(monkeypatch, platform, env=None):
    monkeypatch.setattr(sys, "platform", platform)
    for k in ("OMP_NUM_THREADS", "KMP_DUPLICATE_LIB_OK"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    importlib.reload(silent_rag)
    import os
    return os.environ.get("OMP_NUM_THREADS"), os.environ.get("KMP_DUPLICATE_LIB_OK")


def test_macos_gets_one_openmp_thread(monkeypatch):
    assert reload_on(monkeypatch, "darwin") == ("1", "TRUE")


def test_an_explicit_setting_wins(monkeypatch):
    assert reload_on(monkeypatch, "darwin", {"OMP_NUM_THREADS": "4"})[0] == "4"


def test_other_platforms_are_left_alone(monkeypatch):
    assert reload_on(monkeypatch, "linux") == (None, None)
