"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { asset, f3 } from "@/lib/format";

function find(all, q) {
  q = q.trim().toLowerCase();
  if (!q) return null;
  return all.find((x) => x.q.toLowerCase() === q)
    || all.find((x) => x.q.toLowerCase().startsWith(q))
    || all.find((x) => x.q.toLowerCase().includes(q));
}

function Pane({ title, r }) {
  return (
    <div className="pane">
      <div className="pane__head">
        <span className="muted">{title}</span>
        <span className="muted">nDCG@10 <b>{f3(r.ndcg)}</b></span>
      </div>
      <ol>
        {r.r.map(([t, rel]) => (
          <li key={t}>
            <span className="results__title" title={t}>{t}</span>
            <span className={`rel ${rel ? "" : "rel--no"}`}>{rel ? "relevant" : "not relevant"}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

// Every test query's change in nDCG@10 at 50%, sorted; the current query is drawn in white.
function Spread({ deltas, hit }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height, mid = H / 2;
    const bw = W / deltas.length;
    const top = Math.max(...deltas.map((e) => Math.abs(e.d)));
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#273047";
    ctx.fillRect(0, mid, W, 1);
    deltas.forEach((e, i) => {
      const h = (e.d / top) * (mid - 8);
      const me = e.x === hit;
      ctx.fillStyle = me ? "#e6e9ef" : e.d < 0 ? "#6ea8fe" : e.d > 0 ? "#56c596" : "#273047";
      ctx.fillRect(i * bw, h < 0 ? mid : mid - h, me ? Math.max(bw, 4) : Math.max(bw - 0.6, 1), Math.max(Math.abs(h), me ? 6 : 1));
    });
  }, [deltas, hit]);
  return <canvas ref={ref} width="900" height="300" role="img" aria-label="Change in nDCG@10 for each of the 323 test queries with half the index upgraded, sorted from largest drop to largest gain" />;
}

export default function QueryExplorer() {
  const [all, setAll] = useState(null);
  const [q, setQ] = useState("domoic acid");

  useEffect(() => {
    fetch(asset("/data/queries.json")).then((r) => r.json()).then(setAll);
  }, []);

  const deltas = useMemo(
    () => (all ? all.map((x) => ({ x, d: x.mixed.ndcg - x.clean.ndcg })).sort((a, b) => a.d - b.d) : []),
    [all],
  );
  const picks = useMemo(
    () => (all ? all.filter((x) => x.clean.ndcg - x.mixed.ndcg > 0.3 && x.q.length < 22).slice(0, 3).map((x) => x.q) : []),
    [all],
  );
  const hit = all ? find(all, q) : null;
  const worse = deltas.filter((e) => e.d < -1e-9).length;
  const better = deltas.filter((e) => e.d > 1e-9).length;

  return (
    <div className="explorer">
      <div className="explorer__input">
        <label htmlFor="q" className="label">NFCorpus test query</label>
        <input id="q" type="text" value={q} onChange={(e) => setQ(e.target.value)} autoComplete="off" spellCheck="false" />
        <p className="examples">
          <span className="muted">Try</span>
          {picks.map((p) => (
            <button key={p} type="button" onClick={() => setQ(p)}>{p}</button>
          ))}
        </p>
      </div>

      <div className="explorer__answer" aria-live="polite">
        {!all ? (
          <p className="muted">Loading queries…</p>
        ) : !hit ? (
          <p className="muted">No test query matches that. Try a health topic, like “vitamin d” or “sugar”.</p>
        ) : (
          <>
            <p className="muted explorer__match">Showing “{hit.q}”</p>
            <div className="panes" data-ui>
              <Pane title="Clean index" r={hit.clean} />
              <Pane title="Half the index upgraded" r={hit.mixed} />
            </div>
          </>
        )}
      </div>

      <div className="figure spread" data-ui>
        <div className="spread__head">
          <span>All 323 test queries, half the index upgraded</span>
          {all && <span className="muted num"><b>{worse}</b> worse · <b>{deltas.length - worse - better}</b> unchanged · <b>{better}</b> better</span>}
        </div>
        <Spread deltas={deltas} hit={hit} />
        <p className="muted small">Each bar is one query’s change in nDCG@10, sorted. The white bar is the query above.</p>
      </div>
    </div>
  );
}
