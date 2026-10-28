import summary from "@/data/summary.json";
import Monitor from "@/components/Monitor";
import IndexMap from "@/components/IndexMap";
import QueryExplorer from "@/components/QueryExplorer";
import DropChart from "@/components/DropChart";
import Reveal from "@/components/Reveal";
import { f3 } from "@/lib/format";

const REPO = "https://github.com/Srikarmk/Silent-RAG-Failures";

const steps = [
  {
    title: "Sample a window",
    text: "100 NFCorpus test queries, with a separate 100 as the clean reference.",
    rows: [["Queries", "100 of 323"], ["Reference", "separate 100"]],
    tag: ["sampled", ""],
  },
  {
    title: "Break the index",
    text: "Inject one failure at a set severity, here F1 at 30%.",
    rows: [["Failure", "F1, 30%"], ["Swapped", "1,089 of 3,633"]],
    tag: ["damaged", "warn"],
  },
  {
    title: "Score from logs",
    text: "Detectors see queries, top-10 results and the index. Never labels.",
    rows: [["Re-embed", "58 / 200 off"], ["Top-1 mean", "0.505"]],
    tag: ["scored", "warn"],
  },
  {
    title: "Grade with labels",
    text: "A paired bootstrap on nDCG@10 decides whether the damage was real.",
    rows: [["nDCG@10 drop", "0.032"], ["p-value", "0.001"]],
    tag: ["harmful", "bad"],
  },
];

const failures = [
  ["F1", "Partial embedding upgrade", "two models’ vectors in one index"],
  ["F2", "Stale documents", "outdated content next to its replacement"],
  ["F3", "Chunking change", "part of the corpus re-chunked"],
  ["F4", "Query shift", "traffic moves to topics the corpus lacks"],
  ["F5", "Config mismatch", "index and queries encoded differently"],
];

const detectors = [
  ["Re-embedding check", "Re-encodes a sample of documents and compares them with the stored vectors", "AUROC 1.000", "Built"],
  ["Retrieved-document classifier", "Separates retrieved documents from the reference window’s", "AUROC 0.907", "Built"],
  ["Query classifier", "Same test on query embeddings; F1 leaves queries alone, so it should not fire", "AUROC 0.510", "Built"],
  ["PSI and KS", "Distribution shift in retrieval scores such as the top-1 similarity", "–", "Nov 11"],
  ["MMD", "Two-sample test on query and document embeddings", "–", "Nov 11"],
  ["Canary queries", "Fixed probe queries whose answers are known", "–", "Nov 18"],
  ["LLM judge", "An LLM rates the retrieved passages, with cost per check", "–", "Nov 18"],
];

const milestones = [
  ["Oct 18", "Baseline: model, truncation, clean CIs", "Done", "done"],
  ["Oct 31", "Failure injection: F1–F5 and negative controls", "In progress", "now"],
  ["Nov 11", "Drift and score-based detectors, tuned on NFCorpus", "Next", ""],
  ["Nov 18", "Canary queries, re-embedding, LLM judge, cost per check", "Planned", ""],
  ["Nov 24", "Held-out run on SciFact and FiQA, detector heatmap", "Planned", ""],
  ["Dec 1", "Benchmark release, monitoring guide, report", "Planned", ""],
];

export default function Page() {
  const last = summary.levels[summary.levels.length - 1];
  const drop = Math.round((1 - last.ndcg / summary.levels[0].ndcg) * 100);

  return (
    <>
      <header className="bar">
        <div className="wrap bar__inner">
          <a className="wordmark" href="#top">Silent Failures</a>
          <nav aria-label="Sections">
            <a href="#how">How it works</a>
            <a href="#failures">Failures</a>
            <a href="#detectors">Detectors</a>
            <a href="#status">Status</a>
            <a className="bar__cta" href={REPO}>GitHub</a>
          </nav>
        </div>
      </header>

      <main id="top" className="wrap">
        <section className="hero" aria-labelledby="hero-title">
          <h1 id="hero-title">No errors. Worse answers.</h1>
          <p className="lead">
            Silent Failures is a benchmark that breaks RAG retrieval on purpose, then tests which
            label-free signals notice before users do.
          </p>
          <Monitor levels={summary.levels} examples={summary.examples.slice(0, 3)} />
          <p className="note">
            Real NFCorpus numbers from a prototype of F1: part of the index re-embedded with
            all-MiniLM-L6-v2 while queries stay on multi-qa-MiniLM-L6-cos-v1.
          </p>
        </section>

        <Reveal as="section" className="section" id="how" aria-labelledby="how-title">
          <h2 id="how-title">How a check runs</h2>
          <div className="steps">
            {steps.map((s) => (
              <div className="step" key={s.title}>
                <div className="ticket" data-ui>
                  <div className="ticket__head">
                    <span>Window 7</span>
                    <span className={`tag ${s.tag[1] && `tag--${s.tag[1]}`}`}>{s.tag[0]}</span>
                  </div>
                  <dl>
                    {s.rows.map(([k, v]) => (
                      <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
                    ))}
                  </dl>
                </div>
                <h3>{s.title}</h3>
                <p>{s.text}</p>
              </div>
            ))}
          </div>
        </Reveal>

        <Reveal as="section" className="section split" aria-labelledby="map-title">
          <div className="split__text">
            <h2 id="map-title">Same text, new vectors</h2>
            <p>
              Each line is one NFCorpus document, from where the index model put it to where
              all-MiniLM-L6-v2 puts the same text. Both models output 384 dimensions, so nothing
              complains when they are mixed, yet a document’s two vectors share a cosine of only{" "}
              <strong>{summary.map_cos.toFixed(2)}</strong> on average.
            </p>
            <p className="note">All 3,633 documents on the clean index’s first two principal components; lines show 260 of them.</p>
          </div>
          <IndexMap />
        </Reveal>

        <Reveal as="section" className="section" id="failures" aria-labelledby="failures-title">
          <h2 id="failures-title">Five ways to break it</h2>
          <p className="section__lead">
            Each failure runs at 10% to 90% of the corpus or query traffic. Harmless changes run
            alongside to measure false alarms.
          </p>
          <div className="cols">
            <ul className="list">
              {failures.map(([id, name, how]) => (
                <li key={id}>
                  <span className="list__id">{id}</span>
                  <span><span className="list__name">{name}</span>, {how}</span>
                </li>
              ))}
            </ul>
            <div className="facts">
              <div>
                <h3>Harmless changes</h3>
                <p>Reordering index rows and mixing 256- and 512-token truncation are the candidates. The final set is chosen in issue #9.</p>
              </div>
              <div>
                <h3>Datasets</h3>
                <p>NFCorpus (3,633 medical documents) for all tuning. SciFact (5,183 abstracts) and FiQA (57,638 forum posts) are held out until the final run.</p>
              </div>
              <div>
                <h3>What counts as harm</h3>
                <p>A window is harmful when its nDCG@10 drop is significant under a one-sided paired bootstrap at α = 0.05. Detectors never see this label.</p>
              </div>
            </div>
          </div>
        </Reveal>

        <Reveal as="section" className="section" id="detectors" aria-labelledby="detectors-title">
          <h2 id="detectors-title">Detectors</h2>
          <p className="section__lead">Each one sees only what a production system logs. Results so far are from the prototype F1 run.</p>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th scope="col">Detector</th><th scope="col">Signal</th><th scope="col">Prototype F1</th><th scope="col">Status</th></tr>
              </thead>
              <tbody>
                {detectors.map(([name, signal, result, status]) => (
                  <tr key={name}>
                    <td className="table__name">{name}</td>
                    <td className="muted">{signal}</td>
                    <td className="num">{result}</td>
                    <td className={status === "Built" ? "good" : "muted"}>{status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Reveal>

        <Reveal as="section" className="section split split--chart" aria-labelledby="drop-title">
          <div className="split__text">
            <h2 id="drop-title">{drop}% of nDCG@10 gone, zero errors</h2>
            <p>
              Mean nDCG@10 over all 323 test queries falls from {f3(summary.levels[0].ndcg)} to{" "}
              {f3(last.ndcg)} as more of the index is upgraded. At 10% the drop on a 100-query
              window is only 0.013, and the paired bootstrap already calls it significant
              (p = 0.025).
            </p>
          </div>
          <DropChart levels={summary.levels} />
        </Reveal>

        <Reveal as="section" className="section" aria-labelledby="try-title">
          <h2 id="try-title">Try a query</h2>
          <p className="section__lead">Type any NFCorpus test query to compare its top results on the clean index and with half the index upgraded.</p>
          <QueryExplorer />
        </Reveal>

        <Reveal as="section" className="section split" id="status" aria-labelledby="status-title">
          <div className="split__text">
            <h2 id="status-title">Where the project stands</h2>
            <p>
              The pipeline, the experiment runner and the baseline are done. A window of 100
              queries lets the paired bootstrap catch an nDCG@10 drop of about 0.06 in 80% of
              windows. Once every injector and detector is in, everything runs once on SciFact
              and FiQA, and the detector-by-failure heatmap replaces this section.
            </p>
            <table className="table table--compact">
              <caption>Clean nDCG@10, multi-qa-MiniLM-L6-cos-v1, 256 tokens</caption>
              <thead><tr><th scope="col">Dataset</th><th scope="col">Queries</th><th scope="col">nDCG@10</th><th scope="col">95% CI</th></tr></thead>
              <tbody>
                <tr><td>NFCorpus</td><td className="num">323</td><td className="num">0.292</td><td className="num">0.259–0.325</td></tr>
                <tr><td>SciFact</td><td className="num">300</td><td className="num">0.546</td><td className="num">0.496–0.595</td></tr>
                <tr><td>FiQA</td><td className="num">648</td><td className="num">0.360</td><td className="num">0.332–0.387</td></tr>
              </tbody>
            </table>
          </div>
          <ol className="milestones">
            {milestones.map(([date, name, state, kind]) => (
              <li key={date} className={kind && `is-${kind}`}>
                <span className="muted num">{date}</span>
                <span>{name}</span>
                <span className="milestones__state">{state}</span>
              </li>
            ))}
          </ol>
        </Reveal>
      </main>

      <footer className="footer">
        <div className="wrap footer__inner">
          <div>
            <p className="wordmark">Silent Failures</p>
            <p className="muted">DS5500 Capstone, Northeastern University, Fall 2026</p>
          </div>
          <p className="muted">Srikar Manda, Alekhya Bulusu, Shane Hussey, Ying-Jen Chiang</p>
          <p>
            <a href={REPO}>Code</a>
            <a href={`${REPO}/issues`}>Issues</a>
          </p>
        </div>
      </footer>
    </>
  );
}
