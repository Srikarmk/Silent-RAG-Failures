from pathlib import Path

import pytest
import yaml

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.detect.base import Detector
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.inject.base import Injector

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_configs_are_valid(path):
    cfg = yaml.safe_load(path.read_text())
    runner.check_datasets(cfg["datasets"], final=True)
    runner.check_severities(cfg.get("failures") or {})
    assert cfg["window_size"] > 0 and cfg["trials"] > 0


def test_list_prints_registries(monkeypatch, capsys):
    class Inj(Injector):
        name = "test_inj"

        def apply(self, state, severity, seed):
            return state

    class Det(Detector):
        name = "test_det"

        def fit(self, reference):
            pass

        def score(self, window):
            return 0.0

    monkeypatch.setitem(INJECTORS, Inj.name, Inj)
    monkeypatch.setitem(DETECTORS, Det.name, Det)
    monkeypatch.setattr("sys.argv", ["run", "--list"])
    runner.main()
    out = capsys.readouterr().out
    assert "test_inj" in out and "test_det" in out


def test_dev_and_final_use_the_same_retriever():
    """Whatever we tune on NFCorpus has to be what runs on the held-out sets."""
    dev, final = (yaml.safe_load((CONFIGS[0].parent / f"{n}.yaml").read_text()) for n in ("dev", "final"))
    for key in ("model", "retriever", "window_size"):
        assert dev[key] == final[key], key
    assert dev["retriever"]["max_seq_length"] == 256
