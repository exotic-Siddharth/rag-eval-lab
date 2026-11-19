"""Browser demo: a single-page UI over the pipeline (`python -m raglab demo`).

Retrieval and generation are separate calls so the page can show the reranked passages in about a second
and the answer when the (slower) reader finishes. Example questions come from the test split, so the page
can show SQuAD's expected answer next to the system's.
"""

from __future__ import annotations

import random
import threading
from importlib import resources

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .data import Question, sample, split_by_document
from .generators import NO_ANSWER, build_generator
from .pipeline import Hit, RAGPipeline, build_pipeline, load_config

ABSTAIN_REASONS = {
    "prompt_injection": "The question looks like a prompt-injection attempt, so it was blocked before retrieval.",
    "empty_query": "The question is empty.",
    "query_too_long": "The question is too long.",
    "low_confidence": "The best passage scored below the confidence gate.",
    "model_abstained": "The reader decided the passages do not contain the answer.",
    "refusal": "The reader refused to answer.",
    "ungrounded": "The reader's answer was not supported by the passages, so the grounding check blocked it.",
}


class RetrieveRequest(BaseModel):
    question: str = Field(..., max_length=1000)


class HitIn(BaseModel):
    chunk_id: str
    score: float


class AnswerRequest(BaseModel):
    question: str = Field(..., max_length=1000)
    hits: list[HitIn] = Field(..., max_length=100)
    generator: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)


def _example(q: Question) -> dict:
    return {"question": q.question, "answers": sorted(set(q.answers)), "answerable": q.answerable}


def create_demo_app(config: str | dict, generators: tuple[str, ...] = ("local", "extractive"),
                    warm_up: bool = True, seed: int = 0) -> FastAPI:
    cfg = load_config(config) if isinstance(config, str) else config
    specs = cfg.get("generators", {})
    names = [g for g in generators if g in specs] or ["extractive"]
    base, docs, questions, cache = build_pipeline(cfg, specs.get(names[0], {"type": "extractive"}))
    pipes: dict[str, RAGPipeline] = {names[0]: base}
    for name in names[1:]:
        pipes[name] = RAGPipeline(base.chunks, base.retriever, base.reranker, build_generator(specs[name], cache),
                                  base.guardrails, base.retrieve_k, base.context_k)
    by_id = {c.chunk_id: c for c in base.chunks}
    titles = sorted({d.title for d in docs})
    gold = {q.question: _example(q) for q in questions}

    ev = cfg.get("eval", {})
    _, test = split_by_document(questions, ev.get("tune_frac", 0.3), seed)
    pool = [_example(q) for q in sample(test, 300, 100, seed)]
    lock = threading.Lock()  # one request at a time: the reader saturates a small CPU on its own

    if warm_up:  # download and load every model now, not on the first visitor's request
        for p in pipes.values():
            p.ask(questions[0].question if questions else "warm-up query")

    app = FastAPI(title="raglab demo", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return resources.files("raglab").joinpath("static/demo.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(base.chunks)}

    @app.get("/api/info")
    def info() -> dict:
        return {
            "documents": len(docs), "chunks": len(base.chunks), "titles": titles,
            "retriever": base.retriever.name, "reranker": base.reranker.name if base.reranker else None,
            "generators": [{"key": k, "name": p.generator.name} for k, p in pipes.items()],
            "context_k": base.context_k, "retrieve_k": base.retrieve_k,
        }

    @app.get("/api/examples")
    def examples(n: int = 4) -> dict:
        n = max(1, min(n, 8))
        rng = random.Random()
        answerable = [e for e in pool if e["answerable"]]
        unanswerable = [e for e in pool if not e["answerable"]]
        picked = rng.sample(answerable, min(n - 1, len(answerable))) + rng.sample(unanswerable, min(1, len(unanswerable)))
        return {"examples": picked}

    @app.post("/api/retrieve")
    def retrieve(req: RetrieveRequest) -> dict:
        g = base.guardrails
        if reason := g.check_query(req.question):
            return {"blocked": reason, "explanation": ABSTAIN_REASONS.get(reason, reason), "passages": []}
        latency: dict = {}
        with lock:
            hits = base.retrieve_many([req.question], latency)[0]
        kept, dropped = g.filter_chunks(hits)
        passages = [{"chunk_id": h.chunk.chunk_id, "title": h.chunk.title, "text": h.chunk.text,
                     "score": round(h.score, 4), "context": i < base.context_k} for i, h in enumerate(kept)]
        return {"blocked": None, "passages": passages[:10], "hits": [{"chunk_id": h.chunk.chunk_id, "score": h.score} for h in kept],
                "dropped_injected": dropped, "latency_ms": latency, "expected": gold.get(req.question)}

    @app.post("/api/answer")
    def answer(req: AnswerRequest) -> dict:
        p = pipes.get(req.generator or names[0])
        if p is None:
            raise HTTPException(400, f"unknown generator; choose from {list(pipes)}")
        if p.guardrails.check_query(req.question):
            raise HTTPException(400, "question was blocked by the input guardrail")
        try:
            hits = [Hit(by_id[h.chunk_id], h.score) for h in req.hits[: p.retrieve_k]]
        except KeyError as e:
            raise HTTPException(400, f"unknown chunk id {e.args[0]}") from None
        latency = {k: float(v) for k, v in req.latency_ms.items() if k in ("retrieve_ms", "rerank_ms")}
        with lock:
            ans = p.answer(req.question, hits, latency)
        ans.latency_ms["total_ms"] = sum(v for k, v in ans.latency_ms.items() if k != "total_ms")
        out = ans.to_dict()
        out["generator"] = p.generator.name
        out["explanation"] = None if not ans.abstained else ABSTAIN_REASONS.get(ans.reason, ans.reason)
        out["no_answer_token"] = NO_ANSWER
        return out

    return app
