"""Retrieval, answer and abstention metrics."""

from __future__ import annotations

import collections
import math

import numpy as np

from .text import normalize_answer


def recall_at_k(rels: list[list[bool]], k: int) -> float:
    """Share of questions with at least one relevant chunk in the top k."""
    return float(np.mean([any(r[:k]) for r in rels])) if rels else 0.0


def mrr_at_k(rels: list[list[bool]], k: int) -> float:
    vals = [next((1.0 / (i + 1) for i, x in enumerate(r[:k]) if x), 0.0) for r in rels]
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k(rels: list[list[bool]], n_relevant: list[int], k: int) -> float:
    vals = []
    for r, n in zip(rels, n_relevant):
        dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(r[:k]) if x)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n, k)))
        vals.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def exact_match(pred: str, golds: list[str]) -> float:
    golds = golds or [""]
    return float(max(normalize_answer(pred) == normalize_answer(g) for g in golds))


def f1_score(pred: str, golds: list[str]) -> float:
    """SQuAD 2.0 token F1; an empty prediction is correct exactly when there is no gold answer."""
    golds = golds or [""]
    best = 0.0
    p_toks = normalize_answer(pred).split()
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not p_toks or not g_toks:
            best = max(best, float(p_toks == g_toks))
            continue
        common = sum((collections.Counter(p_toks) & collections.Counter(g_toks)).values())
        if common:
            prec, rec = common / len(p_toks), common / len(g_toks)
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def auroc(scores: list[float], labels: list[bool]) -> float:
    """Probability that a random positive outscores a random negative (ties count half)."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_for_pass_rate(scores: list[float], pass_rate: float) -> float:
    """Score threshold that lets `pass_rate` of the given (answerable) questions through."""
    return float(np.quantile(np.asarray(scores), 1 - pass_rate))
