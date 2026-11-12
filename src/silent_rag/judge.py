"""LLM relevance judgements (#26), cached so a rerun never pays twice.

One judgement is one (question, document) pair: does the document help answer the
question? The `llm_judge` detector scores a window by the share of judged pairs that
come back irrelevant. Judgements are keyed by model, prompt version and the exact
texts, so the same pair seen under different failures, seeds or windows is judged
once, and changing the prompt or model never reuses an old answer.

Cost: every judgement stores its token usage and dollar cost. Prices are per million
tokens from Anthropic's price list (thinking tokens bill as output); the Batches API
halves them. `agreement` checks the judge against NFCorpus qrels on clean retrieval,
which #26 requires before the judge is trusted as a reference.

    python -m silent_rag.judge agreement --config configs/dev.yaml --model ollama:qwen3:30b-a3b
    python -m silent_rag.judge agreement --config configs/dev.yaml --model claude-haiku-5-5 --estimate
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from silent_rag import pipeline

# v2: v1 called ~55% of clean top-5 results relevant against ~33% in the qrels. v2 asks for a
# specific connection, measured on 80 clean NFCorpus pairs (qwen3:30b-a3b: kappa 0.40 -> 0.47)
PROMPT_VERSION = "v2"
SYSTEM = (
    "You judge results from a search engine over medical and nutrition research abstracts. "
    "You get a user's question and one retrieved document. The document is relevant only if it "
    "directly studies or reports on the specific subject of the question, so that an article "
    "answering the question would cite it as evidence. Sharing a broad topic (nutrition, a disease, "
    "a food group) is not enough. Most retrieved documents are not relevant; say relevant only "
    "when the connection is specific."
)
SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}},
    "required": ["relevant"],
    "additionalProperties": False,
}
# $ per million tokens (input, output)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
# free and local; any PRICES model works too, and costs money
DEFAULT_MODEL = "ollama:qwen3:30b-a3b"
# relevance is a classification call: low effort is the documented setting for those
DEFAULT_EFFORT = "low"
# "ollama:<name>" runs a local model through Ollama (http://localhost:11434): free, no API key
OLLAMA = "ollama:"
OLLAMA_URL = "http://localhost:11434/api/chat"


def is_local(model: str) -> bool:
    return model.startswith(OLLAMA)


def known(model: str) -> bool:
    return model in PRICES or (is_local(model) and len(model) > len(OLLAMA))


@dataclass
class Judgement:
    relevant: bool | None  # None: the model declined or the answer couldn't be parsed
    input_tokens: int
    output_tokens: int
    cost_usd: float
    seconds: float | None = None  # how long the request took; None for batch results and older cache rows


def key(model: str, query: str, doc: str) -> str:
    return hashlib.sha1("\0".join([model, PROMPT_VERSION, query, doc]).encode()).hexdigest()


def cost(model: str, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
    if is_local(model):
        return 0.0
    if model not in PRICES:
        raise ValueError(f"no price for {model!r}; add it to PRICES from the published price list")
    p_in, p_out = PRICES[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1e6 * (0.5 if batch else 1.0)


def params(model: str, query: str, doc: str, effort: str = DEFAULT_EFFORT) -> dict:
    """The Messages API request for one pair. The question and document go in the user
    turn; the system prompt stays fixed so every request shares it."""
    return {
        "model": model,
        "max_tokens": 2048,
        "system": SYSTEM,
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": f"<question>\n{query}\n</question>\n\n<document>\n{doc}\n</document>"}],
    }


def parse(message, model: str, batch: bool = False) -> Judgement:
    usage = message.usage
    j = Judgement(None, usage.input_tokens, usage.output_tokens, cost(model, usage.input_tokens, usage.output_tokens, batch))
    if message.stop_reason != "end_turn":  # refusal, max_tokens: no verdict, but the tokens were billed
        return j
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        j.relevant = bool(json.loads(text)["relevant"])
    except (ValueError, KeyError, TypeError):
        pass
    return j


class Cache:
    """sqlite, so parallel runner processes share one store safely."""

    def __init__(self, path: Path | None = None):
        # read at call time, so tests that move the embedding cache move this too
        self.path = Path(path or pipeline.CACHE_DIR.parent / "judge" / "judgements.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS judgements (key TEXT PRIMARY KEY, model TEXT, relevant INTEGER, "
            "input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, created REAL)"
        )
        # added later: what each judgement took, so a cached check can still report its real cost (#27)
        if "seconds" not in {r[1] for r in self.db.execute("PRAGMA table_info(judgements)")}:
            self.db.execute("ALTER TABLE judgements ADD COLUMN seconds REAL")

    def get(self, keys: list[str]) -> dict[str, Judgement]:
        out = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            rows = self.db.execute(
                f"SELECT key, relevant, input_tokens, output_tokens, cost_usd, seconds FROM judgements "
                f"WHERE key IN ({','.join('?' * len(chunk))})", chunk).fetchall()
            for k, rel, ti, to, c, sec in rows:
                out[k] = Judgement(None if rel is None else bool(rel), ti, to, c, sec)
        return out

    def put(self, model: str, items: dict[str, Judgement]) -> None:
        now = time.time()
        self.db.executemany(
            "INSERT OR REPLACE INTO judgements (key, model, relevant, input_tokens, output_tokens, cost_usd, "
            "created, seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(k, model, None if j.relevant is None else int(j.relevant), j.input_tokens, j.output_tokens, j.cost_usd,
              now, j.seconds) for k, j in items.items()])
        self.db.commit()


class OllamaClient:
    """The slice of the Anthropic client judge_live uses (`messages.create`), served by a
    local Ollama model: the same system prompt, user turn and JSON schema, temperature 0,
    thinking off. Returns an object shaped like an Anthropic Message."""

    def __init__(self, url: str = OLLAMA_URL, timeout: float = 600.0):
        self.url, self.timeout = url, timeout
        self.messages = self

    def create(self, model: str, system: str, messages: list, output_config: dict, **_) -> object:
        import urllib.request
        from types import SimpleNamespace

        body = {
            "model": model.removeprefix(OLLAMA),
            "messages": [{"role": "system", "content": system}, *messages],
            "format": output_config["format"]["schema"],
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        return SimpleNamespace(
            stop_reason="end_turn" if out.get("done_reason", "stop") == "stop" else "max_tokens",
            content=[SimpleNamespace(type="text", text=out["message"]["content"])],
            usage=SimpleNamespace(input_tokens=out.get("prompt_eval_count", 0), output_tokens=out.get("eval_count", 0)),
        )


def make_client(model: str = DEFAULT_MODEL):
    if is_local(model):
        return OllamaClient()
    import anthropic  # optional dependency: pip install -e ".[judge]"

    return anthropic.Anthropic()


def judge_live(pairs: list[tuple[str, str]], model: str, cache: Cache, client=None,
               effort: str = DEFAULT_EFFORT) -> dict[str, Judgement]:
    """Judge every pair not already cached, one request each, and cache the answers."""
    keys = [key(model, q, d) for q, d in pairs]
    have = cache.get(keys)
    todo = {k: p for k, p in zip(keys, pairs) if k not in have}
    if todo:
        client = client or make_client(model)
        new = {}
        for k, (q, d) in todo.items():
            start = time.perf_counter()
            reply = client.messages.create(**params(model, q, d, effort))
            new[k] = dataclasses.replace(parse(reply, model), seconds=time.perf_counter() - start)
            if len(new) % 50 == 0:
                cache.put(model, new)
                print(f"judged {len(new)}/{len(todo)}", flush=True)
        cache.put(model, new)
        have.update(new)
    return {k: have[k] for k in keys}


def judge_batch(pairs: list[tuple[str, str]], model: str, cache: Cache, client=None,
                effort: str = DEFAULT_EFFORT, poll_s: float = 60.0) -> dict[str, Judgement]:
    """The same through the Message Batches API, at half price; blocks until it ends."""
    if is_local(model):
        raise ValueError("the Batches API is for Anthropic models; judge local models live (they cost nothing)")
    keys = [key(model, q, d) for q, d in pairs]
    have = cache.get(keys)
    todo = {k: p for k, p in zip(keys, pairs) if k not in have}
    if todo:
        client = client or make_client(model)
        requests = [{"custom_id": k, "params": params(model, q, d, effort)} for k, (q, d) in todo.items()]
        new = {}
        for i in range(0, len(requests), 10_000):
            batch = client.messages.batches.create(requests=requests[i:i + 10_000])
            while client.messages.batches.retrieve(batch.id).processing_status != "ended":
                time.sleep(poll_s)
            for r in client.messages.batches.results(batch.id):
                if r.result.type == "succeeded":
                    new[r.custom_id] = parse(r.result.message, model, batch=True)
            cache.put(model, new)
        have.update(new)
    return {k: have[k] for k in keys if k in have}


def estimate(pairs: list[tuple[str, str]], model: str, output_tokens: int = 150, batch: bool = False) -> dict:
    """A cost estimate before spending anything: input tokens from text length
    (~4 characters a token, plus the fixed prompt), output from a per-call guess."""
    n = len(pairs)
    chars = sum(len(q) + len(d) for q, d in pairs) + n * (len(SYSTEM) + 60)
    tin = chars / 4
    return {"pairs": n, "input_tokens": int(tin), "output_tokens": n * output_tokens,
            "cost_usd": round(cost(model, int(tin), n * output_tokens, batch), 2)}


# agreement with qrels


def clean_pairs(cfg: dict, depth: int = 5):
    """(query id, doc id, query text, doc text) for the top `depth` clean results of every
    query in the config's dev dataset."""
    from silent_rag import data
    from silent_rag import run as runner

    if [runner.canonical(d) for d in cfg["datasets"]] != ["nfcorpus"]:
        raise SystemExit("agreement is checked on NFCorpus only")
    ds = data.load("nfcorpus")
    retriever = runner.default_retriever(cfg)
    state = retriever.embed_corpus(ds)
    ids, _, _ = retriever.search(state, k=depth, query_embeddings=state.query_embeddings)
    out = []
    for qi, row in zip(state.query_ids, ids):
        for d in row:
            out.append((qi, d, ds.queries[qi], ds.corpus[d]))
    return ds, out


def agreement(rows: list[tuple[str, str]], judged: list[bool | None], qrels: dict) -> dict:
    """Judge vs qrels on (query id, doc id) rows: any positive grade counts as relevant."""
    truth = np.array([qrels.get(q, {}).get(d, 0) > 0 for q, d in rows])
    ok = np.array([j is not None for j in judged])
    pred = np.array([bool(j) for j in judged])
    t, p = truth[ok], pred[ok]
    tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
    n = len(t)
    po = (tp + tn) / n if n else float("nan")
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / n**2 if n else float("nan")
    return {"pairs": int(len(rows)), "unjudged": int((~ok).sum()), "accuracy": float(po),
            "kappa": float((po - pe) / (1 - pe)) if pe < 1 else float("nan"),
            "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
            "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "qrels_relevant_share": float(t.mean()) if n else float("nan"),
            "judge_relevant_share": float(p.mean()) if n else float("nan")}


def main() -> None:
    import os

    # FAISS and PyTorch each bring an OpenMP runtime; on macOS more than one thread segfaults
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import yaml

    from silent_rag import run as runner

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("agreement", help="judge clean top-k results and compare with qrels")
    a.add_argument("--config", required=True)
    a.add_argument("--model", default=DEFAULT_MODEL)
    a.add_argument("--effort", default=DEFAULT_EFFORT)
    a.add_argument("--depth", type=int, default=5)
    a.add_argument("--batch", action="store_true", help="use the Batches API (half price, slower)")
    a.add_argument("--estimate", action="store_true", help="print the cost estimate and stop, no API calls")
    args = parser.parse_args()

    cfg = yaml.load(open(args.config), Loader=runner.UniqueKeyLoader)
    ds, rows = clean_pairs(cfg, args.depth)
    pairs = [(qt, dt) for _, _, qt, dt in rows]
    cache = Cache()
    cached = cache.get([key(args.model, q, d) for q, d in pairs])
    print(f"{len(pairs)} pairs, {len(cached)} already cached")
    est = estimate([p for p in pairs if key(args.model, *p) not in cached], args.model, batch=args.batch)
    print(f"estimate for the rest with {args.model}: {est}")
    if args.estimate:
        return
    judge = judge_batch if args.batch else judge_live
    got = judge(pairs, args.model, cache, effort=args.effort)
    judged = [got.get(key(args.model, q, d), Judgement(None, 0, 0, 0.0)).relevant for q, d in pairs]
    spent = sum(j.cost_usd for k, j in got.items() if k not in cached)
    print(json.dumps({**agreement([(q, d) for q, d, _, _ in rows], judged, ds.qrels),
                      "model": args.model, "spent_usd": round(spent, 4)}, indent=2))


if __name__ == "__main__":
    main()
