"""Sparse, dense and hybrid retrievers plus a cross-encoder reranker.

All retrievers share one interface: `fit(chunks)` then `search_many(queries, k)` returning, per query,
a list of (chunk_index, score) sorted by descending score.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from .cache import KVCache, stable_hash
from .chunking import Chunk
from .text import tokenize

Hits = list[tuple[int, float]]


def _top_k(scores: np.ndarray, k: int) -> Hits:
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx]


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class BM25Retriever:
    """Okapi BM25 as a precomputed sparse matrix, so a batch of queries is one sparse matmul."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.name = "bm25"

    def fit(self, chunks: list[Chunk]) -> "BM25Retriever":
        self.vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        lengths = np.zeros(len(chunks))
        for i, c in enumerate(chunks):
            counts = Counter(tokenize(f"{c.title} {c.text}"))
            lengths[i] = sum(counts.values())
            for tok, tf in counts.items():
                rows.append(i)
                cols.append(self.vocab.setdefault(tok, len(self.vocab)))
                tfs.append(tf)
        rows, cols, tfs = np.array(rows), np.array(cols), np.array(tfs, dtype=np.float64)
        df = np.bincount(cols, minlength=len(self.vocab))
        idf = np.log1p((len(chunks) - df + 0.5) / (df + 0.5))
        norm = self.k1 * (1 - self.b + self.b * lengths[rows] / lengths.mean())
        weights = idf[cols] * tfs * (self.k1 + 1) / (tfs + norm)
        self.matrix = sparse.csr_matrix((weights, (rows, cols)), shape=(len(chunks), len(self.vocab)))
        return self

    def _query_matrix(self, queries: list[str]) -> sparse.csr_matrix:
        rows, cols = [], []
        for i, q in enumerate(queries):
            for tok in set(tokenize(q)):
                if tok in self.vocab:
                    rows.append(i)
                    cols.append(self.vocab[tok])
        data = np.ones(len(rows))
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(queries), len(self.vocab)))

    def search_many(self, queries: list[str], k: int) -> list[Hits]:
        scores = (self._query_matrix(queries) @ self.matrix.T).toarray()
        return [_top_k(row, k) for row in scores]


class DenseRetriever:
    """Bi-encoder retrieval with cosine similarity; embeddings are cached per (model, text)."""

    def __init__(self, model: str, query_prefix: str = "", cache: KVCache | None = None,
                 device: str = "auto", batch_size: int = 64, name: str | None = None):
        self.model_name, self.query_prefix = model, query_prefix
        self.cache = cache or KVCache(":memory:")
        self.device, self.batch_size = resolve_device(device), batch_size
        self.name = name or f"dense:{model.split('/')[-1]}"
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        ns = f"emb:{self.model_name}"
        keys = [stable_hash(t) for t in texts]
        cached = self.cache.get_vectors(ns, keys)
        missing = sorted({k: t for k, t in zip(keys, texts) if k not in cached}.items())
        if missing:
            vecs = self.model.encode([t for _, t in missing], batch_size=self.batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True)
            new = {k: v for (k, _), v in zip(missing, vecs)}
            self.cache.put_vectors(ns, new)
            cached.update(new)
        return np.stack([cached[k] for k in keys]).astype(np.float32)

    def fit(self, chunks: list[Chunk]) -> "DenseRetriever":
        self.matrix = self.embed([f"{c.title}. {c.text}" for c in chunks])
        return self

    def search_many(self, queries: list[str], k: int) -> list[Hits]:
        # Queries are embedded fresh (not cached) so measured latency is the real online cost.
        q = self.model.encode([self.query_prefix + q for q in queries], batch_size=self.batch_size,
                              normalize_embeddings=True, convert_to_numpy=True)
        scores = q.astype(np.float32) @ self.matrix.T
        return [_top_k(row, k) for row in scores]


class HybridRetriever:
    """Reciprocal-rank fusion: robust to the very different score scales of BM25 and cosine."""

    def __init__(self, retrievers: list, rrf_k: int = 60, pool: int = 100):
        self.retrievers, self.rrf_k, self.pool = retrievers, rrf_k, pool
        self.name = "hybrid(" + "+".join(r.name for r in retrievers) + ")"

    def fit(self, chunks: list[Chunk]) -> "HybridRetriever":
        for r in self.retrievers:
            r.fit(chunks)
        return self

    def search_many(self, queries: list[str], k: int) -> list[Hits]:
        per_retriever = [r.search_many(queries, self.pool) for r in self.retrievers]
        out = []
        for qi in range(len(queries)):
            fused: dict[int, float] = {}
            for hits in per_retriever:
                for rank, (idx, _) in enumerate(hits[qi]):
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            out.append(sorted(fused.items(), key=lambda x: -x[1])[:k])
        return out


class CrossEncoderReranker:
    """Re-scores the top-N candidates jointly with the query; its logit also drives the abstain gate."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 20,
                 device: str = "auto", batch_size: int = 64, cache: KVCache | None = None):
        self.model_name, self.top_n, self.batch_size = model, top_n, batch_size
        self.device = resolve_device(device)
        self.cache = cache or KVCache(":memory:")
        self.name = f"rerank:{model.split('/')[-1]}@{top_n}"
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        ns = f"ce:{self.model_name}"
        keys = [stable_hash(q, p) for q, p in pairs]
        cached = self.cache.get_vectors(ns, keys)
        todo = [i for i, k in enumerate(keys) if k not in cached]
        if todo:
            scores = self.model.predict([pairs[i] for i in todo], batch_size=self.batch_size,
                                        convert_to_numpy=True, show_progress_bar=False)
            new = {keys[i]: np.array([s], dtype=np.float32) for i, s in zip(todo, scores)}
            self.cache.put_vectors(ns, new)
            cached.update(new)
        return np.array([float(cached[k][0]) for k in keys])

    def rerank_many(self, queries: list[str], candidates: list[Hits], chunks: list[Chunk]) -> list[Hits]:
        pairs, owners = [], []
        for qi, (q, hits) in enumerate(zip(queries, candidates)):
            for idx, _ in hits[: self.top_n]:
                pairs.append((q, f"{chunks[idx].title}. {chunks[idx].text}"))
                owners.append((qi, idx))
        scores = self.score_pairs(pairs) if pairs else np.array([])
        out: list[Hits] = [[] for _ in queries]
        for (qi, idx), s in zip(owners, scores):
            out[qi].append((idx, float(s)))
        return [sorted(h, key=lambda x: -x[1]) for h in out]
