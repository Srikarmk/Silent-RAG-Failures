import { f3, pct } from "@/lib/format";

// Mean nDCG@10 at each F1 severity. The y range starts near the lowest value so the drop
// is readable, and the axis says so.
export default function DropChart({ levels }) {
  const lo = 0.2, hi = Math.max(...levels.map((l) => l.ndcg));
  const label = new Set([0, 1, levels.length - 1]);
  return (
    <div className="figure drop" data-ui role="img" aria-label={`nDCG@10 by severity, from ${f3(levels[0].ndcg)} at 0% to ${f3(levels[levels.length - 1].ndcg)} at ${pct(levels[levels.length - 1].severity)}`}>
      <div className="drop__bars">
        {levels.map((l, i) => (
          <div className={`drop__col ${i === 1 ? "is-accent" : ""}`} key={l.severity} tabIndex={0}>
            <span className={`drop__value ${label.has(i) ? "is-shown" : ""}`}>{f3(l.ndcg)}</span>
            <i style={{ height: `${((l.ndcg - lo) / (hi - lo)) * 100}%` }} />
            <span className="drop__x">{pct(l.severity)}</span>
          </div>
        ))}
      </div>
      <p className="muted small">Share of documents upgraded. The y axis starts at {lo.toFixed(1)}.</p>
    </div>
  );
}
