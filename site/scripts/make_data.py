# Builds the project page's data on NFCorpus with the real F1 injector, using the
# all-MiniLM-L6-v2 entry and retriever settings from configs/dev.yaml.
# Usage, from the repo root:
#   python site/scripts/make_data.py [results.parquet]
# Writes site/data/summary.json (bundled into the page) and site/public/data/{map,queries}.json
# (fetched by the browser). With a results parquet from `python -m silent_rag.run --config
# configs/dev.yaml`, it also writes site/data/benchmark.json, the benchmark tables on the page.
import json, sys
import numpy as np
import yaml
from datasets import load_dataset
from silent_rag.data import load, SOURCES
from silent_rag.inject.f1_partial_upgrade import PartialUpgrade
from silent_rag.pipeline import Retriever
from silent_rag.eval.ground_truth import per_query_ndcg, paired_bootstrap

cfg = yaml.safe_load(open("configs/dev.yaml"))
f1 = {k: v for k, v in cfg["failures"]["F1_partial_upgrade_minilm"].items() if k.startswith("alt_")}
ds = load("nfcorpus")
r = Retriever(cfg["model"], **cfg["retriever"])
state = r.embed_corpus(ds)
qids = list(ds.queries); qtexts = [ds.queries[q] for q in qids]
qemb = r.embed_queries(qtexts)
inj = PartialUpgrade(**f1)
inj.attach(r, state)
alt = inj.alt_embeddings
src = SOURCES["nfcorpus"]
titles = {str(d["_id"]): d["title"] for d in load_dataset(src["repo"], "corpus", split="corpus", revision=src["revision"])}

sevs = [round(0.1 * i, 1) for i in range(10)]
sample = np.random.default_rng(2).choice(len(state.doc_ids), 200, replace=False)
rng = np.random.default_rng(1)
window = rng.choice(len(qids), 100, replace=False)
out = {"n_docs": len(state.doc_ids), "n_queries": len(qids), "levels": []}
clean_nd = None
per_q = {}
for s in sevs:
    # one seed for every severity, so higher severities upgrade a superset of rows
    st = inj.apply(state, s, seed=0) if s else state
    rows = np.flatnonzero((st.doc_embeddings != state.doc_embeddings).any(1))
    ids, scores, _ = r.search(st, k=10, query_embeddings=qemb)
    nd = per_query_ndcg(ids, qids, ds.qrels)
    if clean_nd is None: clean_nd = nd
    w = window
    p = paired_bootstrap(clean_nd[w], nd[w])
    out["levels"].append({
        "severity": s, "ndcg": float(nd.mean()), "ndcg_window": float(nd[w].mean()),
        "drop_window": float(clean_nd[w].mean() - nd[w].mean()),
        "p_value": p[1],
        "top1_mean": float(np.mean(scores[:, 0])), "swapped": int(len(rows)),
        "queries_worse": int((nd < clean_nd - 1e-9).sum()),
        "reembed_mismatch": int(np.isin(sample, rows).sum()),
    })
    per_q[s] = (ids, scores, nd)
    print(s, out["levels"][-1], file=sys.stderr)

# example queries: clean nDCG high and a clear drop by 0.5
cand = [i for i in range(len(qids)) if per_q[0.0][2][i] >= 0.6 and per_q[0.5][2][i] <= 0.3 and len(qtexts[i]) < 60]
examples = []
for i in cand[:6]:
    q = qids[i]
    ex = {"id": q, "text": qtexts[i], "levels": []}
    for s in sevs:
        ids, scores, nd = per_q[s]
        ex["levels"].append({"ndcg": float(nd[i]), "results": [
            {"title": titles.get(d, d)[:90], "score": round(float(sc), 3), "rel": int(ds.qrels[q].get(d, 0))}
            for d, sc in zip(ids[i][:5], scores[i][:5])]})
    examples.append(ex)
out["examples"] = examples
def top3(s, i):
    ids, scores, nd = per_q[s]
    return {"ndcg": round(float(nd[i]), 3), "r": [[titles.get(d, d)[:80], int(ds.qrels[qids[i]].get(d, 0) > 0)] for d in ids[i][:3]]}
out["all"] = [{"q": qtexts[i], "clean": top3(0.0, i), "mixed": top3(0.5, i)} for i in range(len(qids))]

# index map: every doc projected on the clean index's first two principal components,
# plus where a sample of docs lands once the other model re-embeds them
X = state.doc_embeddings
mu = X.mean(0)
P = np.linalg.svd(X - mu, full_matrices=False)[2][:2].T
a, b = (X - mu) @ P, (alt - mu) @ P
lo, hi = a.min(0), a.max(0)
norm = lambda v: np.round((v - lo) / (hi - lo), 3).tolist()
moved = np.random.default_rng(3).choice(len(X), 260, replace=False)
out["map"] = {"docs": norm(a), "from": norm(a[moved]), "to": norm(b[moved]),
              "cos": round(float((X * alt).sum(1).mean()), 2)}
def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))

dump({k: out[k] for k in ("n_docs", "n_queries", "levels", "examples")} | {"map_cos": out["map"]["cos"]},
     "site/data/summary.json")
dump(out["map"], "site/public/data/map.json")
dump(out["all"], "site/public/data/queries.json")

# benchmark tables from a runner results parquet: damage per F1 variant and severity, and per
# detector the AUROC (seed-block CI) and the share of harmful windows caught at each severity,
# with the threshold at the detector's highest score on clean windows (no clean window alarms)
if len(sys.argv) > 1:
    import pandas as pd
    from silent_rag.eval.detector_scoring import bootstrap_auroc_ci, severity_detection_table
    df = pd.read_parquet(sys.argv[1])
    labels = [f for f in cfg["failures"] if f in set(df.failure)]
    win = df.drop_duplicates(["failure", "severity", "seed", "trial"])
    res = {"run_id": df.run_id.iloc[0], "seeds": sorted(int(s) for s in df.seed.unique()),
           "trials": int(df.trial.nunique()), "window": int(df.window_size.iloc[0]),
           "failures": [], "detectors": []}
    for f in labels:
        w = win[win.failure == f]
        res["failures"].append({"label": f, "alt_model": cfg["failures"][f]["alt_model"], "levels": [
            {"severity": float(s), "drop": round(float(g.ndcg_drop.mean()), 3),
             "harmful": round(float(g.harmful.mean()), 2)} for s, g in w.groupby("severity")]})
    for det in cfg["detectors"]:
        thr = df[(df.detector == det) & (df.failure == "clean")].score.max()
        for f in labels:
            auc, lo, hi = bootstrap_auroc_ci(df, det, f)
            t = severity_detection_table(df, det, f, thr)
            res["detectors"].append({"detector": det, "label": f, "auroc": round(auc, 3),
                                     "ci": [round(lo, 3), round(hi, 3)],
                                     "caught": [round(float(v), 2) for v in t.tpr_when_harmful]})
    dump(res, "site/data/benchmark.json")
    print(json.dumps(res, indent=1), file=sys.stderr)
