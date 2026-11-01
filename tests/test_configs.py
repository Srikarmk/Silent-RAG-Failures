from pathlib import Path

import pytest
import yaml

from silent_rag import run as runner
from silent_rag.detect.base import REGISTRY as DETECTORS
from silent_rag.detect.base import Detector
from silent_rag.inject.base import REGISTRY as INJECTORS
from silent_rag.inject.base import Injector

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


def load(path):
    """Parse a config the way main() and run() do."""
    runner.load_plugins()
    cfg = yaml.load(path.read_text(), Loader=runner.UniqueKeyLoader)
    return cfg, runner.failure_params(cfg.get("failures"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_configs_are_valid(path):
    cfg, _ = load(path)
    runner.check_datasets(cfg["datasets"], final=True)
    assert cfg["window_size"] > 0 and cfg["trials"] > 0


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_configs_pass_injector_checks(path):
    """The real configs get past every injector's check_config, with the run_cfg run() builds."""
    cfg, failures = load(path)
    runner.check_injectors(failures, runner.injector_run_cfg(runner.default_retriever(cfg)))


def test_dev_and_final_run_the_same_f1():
    """The held-out run has to damage the index the same way the dev run does.
    Severity grids may differ, so only the injector and its params are compared."""
    dev, final = (
        {label: (name, params) for label, (name, _, params) in load(CONFIGS[0].parent / f"{n}.yaml")[1].items()
         if name == "F1_partial_upgrade"}
        for n in ("dev", "final")
    )
    assert set(dev) == {"F1_partial_upgrade_minilm", "F1_partial_upgrade_bge"}
    assert dev == final


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
    dev, final = (load(CONFIGS[0].parent / f"{n}.yaml")[0] for n in ("dev", "final"))
    for key in ("model", "retriever", "window_size"):
        assert dev[key] == final[key], key
    assert runner.run_seeds(dev) == runner.run_seeds(final), "seeds"
    assert dev["retriever"]["max_seq_length"] == 256
