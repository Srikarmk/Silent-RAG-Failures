import pandas as pd
import pytest

from silent_rag.eval import figures as F
from silent_rag.eval import report as R
from test_report import table


def test_tables_are_the_reports_numbers_one_row_per_detector():
    df = table(trials=20)
    t = F.tables(df, min_drop=0.02, n_boot=100)
    # the good detector `a` first, then `b`
    assert list(t["auroc"].index) == ["a", "b"]
    auc = R.auroc_table(R.relabel(df, 0.02), n_boot=100).set_index("detector")
    assert t["auroc"].loc["a", "F1_x"] == auc.loc["a", "auroc"]
    alarms = R.alarm_table(R.relabel(df, 0.02)).set_index("detector")
    assert t["false_alarm"].loc["b", "H_y"] == alarms.loc["b", "false alarms: H_y"]
    assert t["worst_seed"].loc["b", "clean"] == alarms.loc["b", "worst seed: clean"]
    assert set(t["earliest"].columns) == {"F1_x"}


def test_one_run_per_set_of_figures():
    df = table()
    df.loc[df.seed == 0, "run_id"] = "r2"
    with pytest.raises(SystemExit, match="one run"):
        F.tables(df)


def test_writes_every_figure_and_table(tmp_path, monkeypatch):
    path = tmp_path / "run.parquet"
    table(trials=20).to_parquet(path)
    out = tmp_path / "figs"
    monkeypatch.setattr("sys.argv", ["figures", str(path), "--out", str(out), "--n-boot", "50"])
    F.main()
    for f in ("auroc.png", "false_alarms.png", "earliness.png", "cost_accuracy.png",
              "auroc.csv", "false_alarm.csv", "worst_seed.csv", "earliest.csv"):
        assert (out / f).stat().st_size > 0, f
    assert list(pd.read_csv(out / "auroc.csv", index_col=0).index) == ["a", "b"]


def test_names_and_markers():
    assert F.name("F1_partial_upgrade_bge") == "F1 bge-small" and F.name("F9_new") == "F9_new"
    assert F.marker("ks_top1") == "o" and F.marker("re_embed") == "P" and F.marker("new_thing") == "v"
