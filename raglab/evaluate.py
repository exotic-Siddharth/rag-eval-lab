"""Evaluation harness.

* `run_retrieval_grid`: every chunker x retriever (x reranker) on the same labelled questions.
* `run_e2e`: one full pipeline; guardrail thresholds are chosen on the *tune* split and reported on
  the *test* split, with an ablation of each guardrail.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics as M
from .cache import KVCache
from .chunking import chunk_corpus, chunker_name
from .data import Question, is_relevant, load_corpus, sample, split_by_document
from .guardrails import Guardrails
from .pipeline import build_pipeline, build_reranker, build_retriever
from .telemetry import latency_summary


def _relevance(questions: list[Question], chunks, ranked) -> tuple[list[list[bool]], list[int]]:
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    n_rel = [sum(is_relevant(c, q) for c in (by_doc[q.doc_id] if q.spans else chunks)) for q in questions]
    rels = [[is_relevant(chunks[i], q) for i, _ in hits] for q, hits in zip(questions, ranked)]
    return rels, n_rel


def _retrieval_row(rels, n_rel, ks) -> dict:
    row = {f"R@{k}": M.recall_at_k(rels, k) for k in ks}
    row["MRR@10"] = M.mrr_at_k(rels, 10)
    row["nDCG@10"] = M.ndcg_at_k(rels, n_rel, 10)
    return row


def run_retrieval_grid(cfg: dict, out_dir: str | Path = "results") -> list[dict]:
    cache = KVCache(cfg.get("cache_path", ".cache/raglab.sqlite"))
    device = cfg.get("device", "auto")
    docs, questions = load_corpus(cfg["corpus"])
    qcfg = cfg.get("questions", {})
    tune, test = split_by_document(questions, qcfg.get("tune_frac", 0.3), qcfg.get("seed", 0))
    qs = sample(test, qcfg.get("n_answerable", 1000), 0, qcfg.get("seed", 0))
    texts = [q.question for q in qs]
    ks = cfg.get("k_values", [1, 3, 5, 10, 20])
    retrievers = [build_retriever(spec, cache, device) for spec in cfg["retrievers"]]
    reranker = build_reranker(cfg.get("reranker"), cache, device)
    rerank_on = set(cfg.get("rerank_on", []))

    rows = []
    for cspec in cfg["chunkers"]:
        chunks = chunk_corpus(docs, cspec)
        cname = chunker_name(cspec)
        stats = {"chunker": cname, "n_chunks": len(chunks),
                 "avg_words": float(np.mean([len(c.text.split()) for c in chunks]))}
        for r in retrievers:
            r.fit(chunks)
            t0 = time.perf_counter()
            ranked = r.search_many(texts, max(ks))
            ms = (time.perf_counter() - t0) * 1000 / len(texts)
            rels, n_rel = _relevance(qs, chunks, ranked)
            stats["coverage"] = float(np.mean([n > 0 for n in n_rel]))
            rows.append({**stats, "retriever": r.name, **_retrieval_row(rels, n_rel, ks), "ms/query": ms})
            print(f"[{cname}] {r.name}: R@5={rows[-1]['R@5']:.3f} MRR@10={rows[-1]['MRR@10']:.3f}", flush=True)
            if reranker is not None and r.name in rerank_on:
                t0 = time.perf_counter()
                reranked = reranker.rerank_many(texts, ranked, chunks)
                rr_ms = (time.perf_counter() - t0) * 1000 / len(texts)
                # pad with the un-reranked tail so R@k stays defined for k > top_n
                merged = [h + [x for x in full if x[0] not in {i for i, _ in h}] for h, full in zip(reranked, ranked)]
                rels, n_rel = _relevance(qs, chunks, merged)
                rows.append({**stats, "retriever": f"{r.name} + {reranker.name}",
                             **_retrieval_row(rels, n_rel, ks), "ms/query": ms + rr_ms})
                print(f"[{cname}] {rows[-1]['retriever']}: R@5={rows[-1]['R@5']:.3f}", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"n_questions": len(qs), "split": "test (answerable only)", "corpus_docs": len(docs)}
    (out / "retrieval_grid.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")
    (out / "retrieval_grid.md").write_text(retrieval_markdown(rows, ks, meta), encoding="utf-8")
    return rows


def retrieval_markdown(rows: list[dict], ks: list[int], meta: dict) -> str:
    head = ["chunker", "chunks", "avg words", "answer coverage", "retriever", *[f"R@{k}" for k in ks],
            "MRR@10", "nDCG@10", "ms/query"]
    lines = [f"Retrieval on {meta['n_questions']} {meta['split']} questions over {meta['corpus_docs']} documents.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        vals = [r["chunker"], str(r["n_chunks"]), f"{r['avg_words']:.0f}", f"{r['coverage']:.3f}", r["retriever"],
                *[f"{r[f'R@{k}']:.3f}" for k in ks], f"{r['MRR@10']:.3f}", f"{r['nDCG@10']:.3f}", f"{r['ms/query']:.1f}"]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _score_setting(records: list[dict], tau: float | None, gamma: float | None) -> dict:
    preds, ems, f1s = [], [], []
    for r in records:
        gated = tau is not None and (r["top_score"] is None or r["top_score"] < tau)
        ungrounded = gamma is not None and not r["raw_abstained"] and r["support"] < gamma
        abstain = gated or r["raw_abstained"] or ungrounded
        pred = "" if abstain else r["raw_answer"]
        preds.append((r, abstain, gated))
        ems.append(M.exact_match(pred, r["answers"]))
        f1s.append(M.f1_score(pred, r["answers"]))
    has = [i for i, r in enumerate(records) if r["answers"]]
    no = [i for i, r in enumerate(records) if not r["answers"]]
    answered = [i for i, (_, a, _) in enumerate(preds) if not a]
    hit = [i for i in has if records[i]["context_hit"]]
    miss = [i for i in has if not records[i]["context_hit"]]
    mean = lambda xs, idx: float(np.mean([xs[i] for i in idx])) if idx else float("nan")  # noqa: E731
    return {
        "EM": float(np.mean(ems)), "F1": float(np.mean(f1s)),
        "HasAns_F1": mean(f1s, has), "NoAns_acc": mean([float(a) for _, a, _ in preds], no),
        "answer_rate": len(answered) / len(records),
        "answered_unanswerable": mean([float(not a) for _, a, _ in preds], no),
        "HasAns_F1_ctx_hit": mean(f1s, hit), "HasAns_F1_ctx_miss": mean(f1s, miss),
        "generation_skipped": float(np.mean([g for _, _, g in preds])),
    }


def run_e2e(cfg: dict, out_dir: str | Path = "results", generator_override: dict | None = None,
            tag: str | None = None) -> dict:
    pipe, docs, questions, cache = build_pipeline(cfg, generator_override)
    ecfg = cfg.get("eval", {})
    seed = ecfg.get("seed", 0)
    tune_all, test_all = split_by_document(questions, ecfg.get("tune_frac", 0.3), seed)
    tune = sample(tune_all, ecfg.get("tune", {}).get("n_answerable", 100), ecfg.get("tune", {}).get("n_unanswerable", 100), seed)
    test = sample(test_all, ecfg.get("test", {}).get("n_answerable", 300), ecfg.get("test", {}).get("n_unanswerable", 300), seed)
    no_guard = Guardrails(drop_injected_chunks=pipe.guardrails.drop_injected_chunks)

    def collect(qs: list[Question], label: str) -> list[dict]:
        hits_all = pipe.retrieve_many([q.question for q in qs])
        records, t0 = [], time.perf_counter()
        for i, (q, hits) in enumerate(zip(qs, hits_all)):
            ans = pipe.answer(q.question, hits, guardrails=no_guard)
            ctx = [h.chunk for h in hits[: pipe.context_k]]
            out, _ = pipe.generate(q.question, hits)  # cached: returns the same generation + its true latency
            records.append({
                "qid": q.qid, "question": q.question, "answers": q.answers, "top_score": ans.top_score,
                "raw_answer": "" if ans.abstained else ans.answer, "raw_abstained": ans.abstained,
                "support": ans.support if ans.support is not None else 1.0,
                "context_hit": any(is_relevant(c, q) for c in ctx) if q.answerable else None,
                "gen_latency_ms": out.latency_s * 1000, "input_tokens": out.input_tokens,
                "output_tokens": out.output_tokens, "cost_usd": out.cost_usd,
            })
            if (i + 1) % 25 == 0:
                print(f"  {label}: {i + 1}/{len(qs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return records

    print(f"tune split: {len(tune)} questions", flush=True)
    tune_rec = collect(tune, "tune")
    print(f"test split: {len(test)} questions", flush=True)
    test_rec = collect(test, "test")

    # Choose guardrail thresholds on the tune split only.
    tune_ans_scores = [r["top_score"] for r in tune_rec if r["answers"] and r["top_score"] is not None]
    taus = {f"pass {p:.0%}": M.threshold_for_pass_rate(tune_ans_scores, p) for p in ecfg.get("gate_pass_rates", [0.95, 0.9, 0.8, 0.7, 0.6])}
    gammas = ecfg.get("grounding_thresholds", [0.5, 0.75, 1.0])
    best_tau = max(taus.items(), key=lambda kv: _score_setting(tune_rec, kv[1], None)["F1"])
    best_gamma = max(gammas, key=lambda g: _score_setting(tune_rec, None, g)["F1"])
    best_both = max(((t, g) for t in taus.items() for g in gammas),
                    key=lambda tg: _score_setting(tune_rec, tg[0][1], tg[1])["F1"])
    settings = {
        "reference: abstain on everything": (float("inf"), None),
        "no guardrails": (None, None),
        f"confidence gate ({best_tau[0]})": (best_tau[1], None),
        f"grounding check (>= {best_gamma})": (None, best_gamma),
        f"gate ({best_both[0][0]}) + grounding (>= {best_both[1]})": (best_both[0][1], best_both[1]),
    }
    ablation = {name: _score_setting(test_rec, t, g) for name, (t, g) in settings.items()}

    # Online latency: single-query retrieval (+ rerank) on fresh caches, plus measured generation time.
    lat_qs = [q.question for q in test[: ecfg.get("latency_queries", 50)]]
    if pipe.reranker is not None:
        pipe.reranker.cache = KVCache(":memory:")
    retr_ms = []
    for q in lat_qs:
        t = {}
        pipe.retrieve_many([q], t)
        retr_ms.append(sum(t.values()))
    gen_ms = [r["gen_latency_ms"] for r in test_rec if r["gen_latency_ms"] > 0]
    gate_scores = [r["top_score"] for r in test_rec]
    gate_labels = [bool(r["answers"]) for r in test_rec]
    result = {
        "config": {"chunker": chunker_name(cfg["chunker"]), "retriever": pipe.retriever.name,
                   "reranker": pipe.reranker.name if pipe.reranker else None, "generator": pipe.generator.name,
                   "context_k": pipe.context_k},
        "n_tune": len(tune_rec), "n_test": len(test_rec),
        "gate_auroc_test": M.auroc(gate_scores, gate_labels),
        "chosen": {"min_confidence": best_both[0][1], "gate": best_both[0][0], "grounding_threshold": best_both[1]},
        "ablation_test": ablation,
        "context_recall_test": float(np.mean([r["context_hit"] for r in test_rec if r["answers"]])),
        "latency": {"retrieval_ms": latency_summary(retr_ms), "generation_ms": latency_summary(gen_ms)},
        "tokens_per_query": {"input": float(np.mean([r["input_tokens"] for r in test_rec])),
                             "output": float(np.mean([r["output_tokens"] for r in test_rec]))},
        "cost_per_1k_queries_usd": float(np.mean([r["cost_usd"] for r in test_rec]) * 1000),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = tag or pipe.generator.name.replace(":", "_").replace("/", "_")
    (out / f"e2e_{name}.json").write_text(json.dumps({"result": result, "test_records": test_rec}, indent=2), encoding="utf-8")
    (out / f"e2e_{name}.md").write_text(e2e_markdown(result), encoding="utf-8")
    return result


def e2e_markdown(res: dict) -> str:
    c = res["config"]
    lines = [f"**{c['generator']}** | {c['chunker']} | {c['retriever']} | reranker: {c['reranker']} | "
             f"top-{c['context_k']} chunks in context", "",
             f"Test: {res['n_test']} questions (half unanswerable). Thresholds tuned on {res['n_tune']} tune-split questions.", "",
             "| setting | EM | F1 | HasAns F1 | NoAns acc | answer rate | generations skipped |",
             "|---|---|---|---|---|---|---|"]
    for name, m in res["ablation_test"].items():
        lines.append(f"| {name} | {m['EM']:.3f} | {m['F1']:.3f} | {m['HasAns_F1']:.3f} | {m['NoAns_acc']:.3f} | "
                     f"{m['answer_rate']:.3f} | {m['generation_skipped']:.3f} |")
    base = res["ablation_test"]["no guardrails"]
    lat = res["latency"]
    lines += ["",
              f"- Context recall (gold span inside the {c['context_k']} chunks given to the generator): {res['context_recall_test']:.3f}",
              f"- HasAns F1 when the gold chunk was retrieved: {base['HasAns_F1_ctx_hit']:.3f}; when it was missed: {base['HasAns_F1_ctx_miss']:.3f}",
              f"- Confidence-gate AUROC (answerable vs unanswerable): {res['gate_auroc_test']:.3f}",
              f"- Latency: retrieval p50 {lat['retrieval_ms']['p50_ms']:.0f} ms / p95 {lat['retrieval_ms']['p95_ms']:.0f} ms; "
              f"generation p50 {lat['generation_ms']['p50_ms']:.0f} ms / p95 {lat['generation_ms']['p95_ms']:.0f} ms",
              f"- Tokens per query: {res['tokens_per_query']['input']:.0f} in / {res['tokens_per_query']['output']:.1f} out; "
              f"API cost per 1k queries: ${res['cost_per_1k_queries_usd']:.2f}"]
    return "\n".join(lines) + "\n"
