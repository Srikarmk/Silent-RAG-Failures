"use client";

import { useState } from "react";
import { f3, pct } from "@/lib/format";

// The hero fragment: drag the share of the index re-embedded with the other model and
// watch nDCG@10 fall while everything an operator normally watches stays green.
export default function Monitor({ levels, examples }) {
  const [sev, setSev] = useState(0);
  const [ex, setEx] = useState(0);
  // the state shown before the last change, so rows that just arrived can animate in
  const [prev, setPrev] = useState(null);
  const go = (nextSev, nextEx) => { setPrev({ sev, ex }); setSev(nextSev); setEx(nextEx); };

  const L = levels[sev];
  const base = levels[0];
  const results = examples[ex].levels[sev].results;
  const previous = (prev ? examples[prev.ex].levels[prev.sev].results : results).map((r) => r.title);
  const delta = Math.round(((base.ndcg - L.ndcg) / base.ndcg) * 100);

  return (
    <div className="monitor" data-ui>
      <div className="monitor__status">
        <span className="ok">Index healthy</span>
        <span><b>3,633</b> docs</span>
        <span><b>0</b> errors</span>
        <span><b>323 / 323</b> queries answered</span>
      </div>

      <label className="monitor__control">
        <span className="label">Partial embedding upgrade (F1): share of documents re-embedded with a different model</span>
        <span className="monitor__slider">
          <input
            type="range" min="0" max={levels.length - 1} step="1" value={sev}
            onChange={(e) => go(Number(e.target.value), ex)}
          />
          <output>{pct(L.severity)}</output>
        </span>
      </label>

      <div className="monitor__query">
        <span className="label">Query</span>
        <div className="tabs" role="tablist">
          {examples.map((e, i) => (
            <button key={e.id} type="button" role="tab" aria-selected={i === ex} onClick={() => go(sev, i)}>
              {e.text}
            </button>
          ))}
        </div>
      </div>

      <ol className="results">
        {results.map((r, i) => (
          <li key={r.title} className={previous.includes(r.title) ? "" : "is-new"}>
            <span className="muted num">{i + 1}</span>
            <span className="results__title" title={r.title}>{r.title}</span>
            <span className="muted num results__score">{r.score.toFixed(3)}</span>
            <span className={`rel ${r.rel ? "" : "rel--no"}`}>{r.rel ? "relevant" : "not relevant"}</span>
          </li>
        ))}
      </ol>

      <div className="monitor__foot">
        <div className="readout">
          <span className="label">nDCG@10, all 323 queries (needs labels)</span>
          <span><b>{f3(L.ndcg)}</b>{sev > 0 && <span className="delta">−{delta}%</span>}</span>
        </div>
        <div className="readout">
          <span className="label">Mean top-1 score (label-free)</span>
          <span><b>{f3(L.top1_mean)}</b></span>
        </div>
        <div className="readout">
          <span className="label">Re-embed check, 200 sampled docs</span>
          <span><b>{L.reembed_mismatch}</b> <span className="muted">mismatched</span></span>
        </div>
      </div>
    </div>
  );
}
