"""FastAPI service exposing the pipeline with per-request telemetry."""

from __future__ import annotations

import threading
from collections import Counter, deque

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import build_pipeline, load_config
from .telemetry import latency_summary


class AskRequest(BaseModel):
    question: str = Field(..., examples=["What is the capital of Normandy?"])


def create_app(config: str | dict, pipeline=None) -> FastAPI:
    cfg = load_config(config) if isinstance(config, str) else config
    pipe = pipeline or build_pipeline(cfg)[0]
    pipe.ask("warm-up query")  # load every model before the first real request
    app = FastAPI(title="raglab", description="Retrieval-augmented QA with guardrails and telemetry")
    lock = threading.Lock()
    recent_ms: deque[float] = deque(maxlen=1000)
    stats = {"requests": 0, "abstained": 0, "cost_usd": 0.0, "llm_cache_hits": 0}
    reasons: Counter[str] = Counter()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(pipe.chunks), "retriever": pipe.retriever.name,
                "reranker": pipe.reranker.name if pipe.reranker else None, "generator": pipe.generator.name}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        ans = pipe.ask(req.question)
        with lock:
            stats["requests"] += 1
            stats["abstained"] += ans.abstained
            stats["cost_usd"] += ans.cost_usd
            stats["llm_cache_hits"] += ans.cached
            reasons[ans.reason] += 1
            recent_ms.append(ans.latency_ms.get("total_ms", 0.0))
        return ans.to_dict()

    @app.get("/stats")
    def get_stats() -> dict:
        with lock:
            return {**stats, "reasons": dict(reasons), "latency": latency_summary(list(recent_ms))}

    return app
