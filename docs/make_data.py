# Builds docs/data.js for the project page: a real NFCorpus run with a prototype F1
# (part of the index re-embedded with all-MiniLM-L6-v2, queries left on the baseline model).
# Usage: python docs/make_data.py docs/data.js
import dataclasses, json, sys
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from silent_rag.data import load, SOURCES
from silent_rag.pipeline import Retriever
from silent_rag.eval.ground_truth import per_query_ndcg, paired_bootstrap

ds = load("nfcorpus")
r = Retriever("sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
              revision="b207367332321f8e44f96e224ef15bc607f4dbf0", max_seq_length=256)
state = r.embed_corpus(ds)
qids = list(ds.queries); qtexts = [ds.queries[q] for q in qids]
qemb = r.embed_queries(qtexts)
alt = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode(
    state.doc_texts, normalize_embeddings=True, batch_size=32).astype(np.float32)
src = SOURCES["nfcorpus"]
titles = {str(d["_id"]): d["title"] for d in load_dataset(src["repo"], "corpus", split="corpus", revision=src["revision"])}

order = np.random.default_rng(0).permutation(len(state.doc_ids))  # nested: higher severity swaps a superset
sevs = [round(0.1 * i, 1) for i in range(10)]
sample = np.random.default_rng(2).choice(len(state.doc_ids), 200, replace=False)
rng = np.random.default_rng(1)
window = rng.choice(len(qids), 100, replace=False)
out = {"n_docs": len(state.doc_ids), "n_queries": len(qids), "levels": []}
clean_nd = None
per_q = {}
for s in sevs:
    emb = state.doc_embeddings.copy()
    rows = order[: int(s * len(order))]
    emb[rows] = alt[rows]
    st = dataclasses.replace(state, doc_embeddings=emb)
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
open(sys.argv[1], "w").write("window.SRF = " + json.dumps(out, separators=(",", ":")) + ";\n")
