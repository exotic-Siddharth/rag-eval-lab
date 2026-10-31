"""The RAG pipeline: retrieve -> rerank -> guard -> generate -> ground-check, with per-stage telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .cache import KVCache
from .chunking import Chunk, chunk_corpus
from .data import load_corpus
from .generators import NO_ANSWER, GenOutput, build_generator
from .guardrails import Guardrails, support_score
from .retrievers import BM25Retriever, CrossEncoderReranker, DenseRetriever, HybridRetriever
from .telemetry import timed


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    reason: str
    citations: list[dict] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    top_score: float | None = None
    support: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    def __init__(self, chunks: list[Chunk], retriever, reranker, generator, guardrails: Guardrails,
                 retrieve_k: int = 20, context_k: int = 4):
        self.chunks, self.retriever, self.reranker = chunks, retriever, reranker
        self.generator, self.guardrails = generator, guardrails
        self.retrieve_k, self.context_k = retrieve_k, context_k

    def retrieve_many(self, questions: list[str], timings: dict | None = None) -> list[list[Hit]]:
        timings = {} if timings is None else timings
        with timed(timings, "retrieve_ms"):
            ranked = self.retriever.search_many(questions, self.retrieve_k)
        if self.reranker is not None:
            with timed(timings, "rerank_ms"):
                ranked = self.reranker.rerank_many(questions, ranked, self.chunks)
        return [[Hit(self.chunks[i], s) for i, s in hits] for hits in ranked]

    def generate(self, question: str, hits: list[Hit]) -> tuple[GenOutput, list[Hit]]:
        contexts = hits[: self.context_k]
        return self.generator.generate(question, [h.chunk for h in contexts]), contexts

    def answer(self, question: str, hits: list[Hit], latency: dict | None = None,
               guardrails: Guardrails | None = None) -> Answer:
        g = guardrails or self.guardrails
        latency = dict(latency or {})
        hits, _ = g.filter_chunks(hits)
        top = hits[0].score if hits else None
        citations = [{"chunk_id": h.chunk.chunk_id, "title": h.chunk.title, "start": h.chunk.start,
                      "end": h.chunk.end, "score": round(h.score, 4)} for h in hits[: self.context_k]]
        if not g.passes_gate(top):
            return Answer(question, NO_ANSWER, True, "low_confidence", citations, latency, top_score=top)
        with timed(latency, "generate_ms"):
            out, contexts = self.generate(question, hits)
        base = dict(citations=citations, latency_ms=latency, input_tokens=out.input_tokens,
                    output_tokens=out.output_tokens, cost_usd=out.cost_usd, cached=out.cached, top_score=top)
        if out.abstained:
            return Answer(question, NO_ANSWER, True, "refusal" if out.refused else "model_abstained", **base)
        support = support_score(out.text, " ".join(h.chunk.text for h in contexts))
        if not g.grounded(out.text, " ".join(h.chunk.text for h in contexts)):
            return Answer(question, NO_ANSWER, True, "ungrounded", support=support, **base)
        return Answer(question, out.text, False, "ok", support=support, **base)

    def ask(self, question: str) -> Answer:
        latency: dict = {}
        if reason := self.guardrails.check_query(question):
            return Answer(question, NO_ANSWER, True, reason)
        hits = self.retrieve_many([question], latency)[0]
        ans = self.answer(question, hits, latency)
        ans.latency_ms["total_ms"] = sum(v for k, v in ans.latency_ms.items() if k != "total_ms")
        return ans


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_retriever(spec: dict, cache: KVCache, device: str):
    kind = spec["type"]
    if kind == "bm25":
        return BM25Retriever(spec.get("k1", 1.2), spec.get("b", 0.75))
    if kind == "dense":
        return DenseRetriever(spec["model"], spec.get("query_prefix", ""), cache, device, name=spec.get("name"))
    if kind == "hybrid":
        return HybridRetriever([build_retriever(m, cache, device) for m in spec["members"]],
                               spec.get("rrf_k", 60), spec.get("pool", 100))
    raise ValueError(f"unknown retriever: {kind}")


def build_reranker(spec: dict | None, cache: KVCache, device: str):
    if not spec:
        return None
    return CrossEncoderReranker(spec.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"), spec.get("top_n", 20),
                                device, cache=cache)


def build_pipeline(cfg: dict, generator_override: dict | None = None):
    cache = KVCache(cfg.get("cache_path", ".cache/raglab.sqlite"))
    device = cfg.get("device", "auto")
    docs, questions = load_corpus(cfg["corpus"])
    chunks = chunk_corpus(docs, cfg["chunker"])
    retriever = build_retriever(cfg["retriever"], cache, device).fit(chunks)
    reranker = build_reranker(cfg.get("reranker"), cache, device)
    generator = build_generator(generator_override or cfg.get("generator", {"type": "extractive"}), cache)
    g = cfg.get("guardrails", {})
    guard = Guardrails(g.get("max_query_chars", 1000), g.get("drop_injected_chunks", True),
                       g.get("min_confidence"), g.get("grounding_threshold"))
    pipe = RAGPipeline(chunks, retriever, reranker, generator, guard, cfg.get("retrieve_k", 20), cfg.get("context_k", 4))
    return pipe, docs, questions, cache
