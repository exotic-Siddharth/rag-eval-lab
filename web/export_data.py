"""Export what the in-browser demo needs, plus the Python pipeline's results for web/parity.mjs.

    python web/export_data.py --config configs/default.yaml --out web/build

Writes chunks.json, examples.json and meta.json (served with the page) and reference.json (used only by
the parity check): for held-out questions, the BM25, dense, fused and reranked rankings, the extractive
answer and the passages that contain the gold answer.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from raglab.data import is_relevant, sample, split_by_document
from raglab.pipeline import Hit, build_pipeline, load_config


def ranked(hits) -> list[list]:
    return [[int(i), round(float(s), 6)] for i, s in hits]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="web/build")
    ap.add_argument("--n-answerable", type=int, default=100)
    ap.add_argument("--n-unanswerable", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipe, docs, questions, _ = build_pipeline(cfg, {"type": "extractive"})
    chunks = pipe.chunks
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, obj) -> None:
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    dump("chunks.json", [{"title": c.title, "text": c.text} for c in chunks])

    _, test = split_by_document(questions, cfg.get("eval", {}).get("tune_frac", 0.3), args.seed)
    pool = sample(test, 300, 100, args.seed)  # same example pool as the server demo
    dump("examples.json", [{"question": q.question, "answers": sorted(set(q.answers)), "answerable": q.answerable}
                           for q in pool])

    hybrid = pipe.retriever
    bm25, dense = hybrid.retrievers
    meta = {
        "documents": len(docs), "chunks": len(chunks), "titles": sorted({d.title for d in docs}),
        "retrieve_k": pipe.retrieve_k, "context_k": pipe.context_k, "pool": hybrid.pool, "rrf_k": hybrid.rrf_k,
        "rerank_top_n": pipe.reranker.top_n, "grounding_threshold": pipe.guardrails.grounding_threshold,
        "python": {"embed_model": dense.model_name, "query_prefix": dense.query_prefix,
                   "rerank_model": pipe.reranker.model_name,
                   "versions": {p: version(p) for p in ("torch", "transformers", "sentence-transformers")}},
    }
    dump("meta.json", meta)

    # The evaluated Python pipeline on held-out test questions (first n of the evaluation's own sample).
    qs = sample(test, args.n_answerable, args.n_unanswerable, args.seed)
    texts = [q.question for q in qs]
    bm25_hits = bm25.search_many(texts, pipe.retrieve_k)
    dense_hits = dense.search_many(texts, pipe.retrieve_k)
    fused = hybrid.search_many(texts, pipe.retrieve_k)
    reranked = pipe.reranker.rerank_many(texts, fused, chunks)
    ref = []
    for q, b, d, f, r in zip(qs, bm25_hits, dense_hits, fused, reranked):
        ans = pipe.answer(q.question, [Hit(chunks[i], s) for i, s in r])
        ref.append({
            "question": q.question, "answerable": q.answerable,
            "relevant": [i for i, c in enumerate(chunks) if q.answerable and is_relevant(c, q)],
            "bm25": ranked(b), "dense": ranked(d), "hybrid": ranked(f), "reranked": ranked(r),
            "answer": ans.answer, "abstained": ans.abstained, "reason": ans.reason,
        })
    dump("reference.json", ref)
    print(f"wrote {len(chunks)} chunks, {len(pool)} examples and {len(ref)} reference questions to {out}")


if __name__ == "__main__":
    main()
