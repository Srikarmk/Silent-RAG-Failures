import os, subprocess, traceback
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import pandas as pd
from silent_rag.plots.style import PlotStyle
from silent_rag.plots import figures_models as fm, figures_statistics as fs

T = Path("results/analysis_f1_v3")
OUT = Path("/tmp/f1_figures"); OUT.mkdir(exist_ok=True)
style = PlotStyle.from_yaml("experiments/analysis_f1/figures.yaml")
t = lambda name: pd.read_parquet(T / f"{name}.parquet")

models = t("replacement_model_difference_table")
drops = t("mean_ndcg10_drop_all_test_questions_by_severity")
stats = t("statistic_all_test_questions_vs_validation_reference_by_severity")
shares = t("share_of_windows_with_significant_ndcg10_drop_by_severity")
seconds = t("median_seconds_per_statistic_evaluation")
auroc_sev = t("auroc_statistic_harmful_vs_negative_windows_by_severity")
auroc_drop = t("auroc_statistic_harmful_vs_negative_windows_by_min_ndcg10_drop")
spearman = t("spearman_statistic_vs_window_mean_ndcg10_drop")
statistics = sorted(stats["statistic"].unique())

jobs = [
    ("models_shared", lambda: fm.replacement_model_difference(models, y="shared", style=style)),
    ("models_switched_ndcg", lambda: fm.replacement_model_difference(models, y="switched_ndcg10", style=style)),
    *[(f"dose_{s}", lambda s=s: fm.dose_response_ndcg10_drop(drops, scale=s, models=models, style=style))
      for s in ("drop", "ndcg", "percent_of_baseline")],
    ("drop_at_1.0_vs_cosine", lambda: fm.ndcg10_drop_at_severity_vs_mean_cosine(drops, models, style=style)),
    *[(f"share_min_drop_{m}", lambda m=m: fs.share_of_significant_windows_vs_severity(shares, min_drop=m, models=models, style=style))
      for m in (0.0, 0.02, 0.06)],
    ("seconds", lambda: fs.seconds_per_statistic(seconds, style=style)),
]
for s in statistics:
    jobs += [
        (f"stat_{s}", lambda s=s: fs.statistic_vs_severity_all_test_questions(stats, statistic=s, models=models, style=style)),
        (f"auroc_sev_{s}", lambda s=s: fs.auroc_vs_severity(auroc_sev, statistic=s, min_drop=0.0, models=models, style=style)),
        (f"auroc_drop_{s}", lambda s=s: fs.auroc_vs_min_ndcg10_drop(auroc_drop, statistic=s, models=models, style=style)),
        (f"spearman_{s}", lambda s=s: fs.spearman_vs_severity(spearman, statistic=s, models=models, style=style)),
    ]

for name, make in jobs:
    try:
        make().savefig(OUT / f"{name}.png", dpi=110)
        print("ok  ", name)
    except Exception:
        print("FAIL", name); traceback.print_exc(limit=1)
subprocess.run(["open", str(OUT)])
