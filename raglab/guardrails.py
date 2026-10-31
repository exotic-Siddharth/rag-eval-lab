"""Guardrails: input validation, prompt-injection filtering, a retrieval-confidence gate and a
post-generation grounding check."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import content_tokens

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|system)",
    r"\byou\s+are\s+now\s+(a|an|in)\b",
    r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)",
    r"<\s*/?\s*(system|instructions?)\s*>",
    r"\bnew\s+instructions\s*:",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def support_score(answer: str, context: str) -> float:
    """Fraction of the answer's content words that occur in the retrieved context (1.0 = fully grounded)."""
    ans = content_tokens(answer)
    if not ans:
        return 1.0
    ctx = set(content_tokens(context))
    return sum(t in ctx for t in ans) / len(ans)


@dataclass
class Guardrails:
    max_query_chars: int = 1000
    drop_injected_chunks: bool = True
    min_confidence: float | None = None  # gate on the top reranker / retriever score
    grounding_threshold: float | None = None  # block answers less supported than this

    def check_query(self, question: str) -> str | None:
        if not question.strip():
            return "empty_query"
        if len(question) > self.max_query_chars:
            return "query_too_long"
        if looks_like_injection(question):
            return "prompt_injection"
        return None

    def filter_chunks(self, hits: list) -> tuple[list, int]:
        if not self.drop_injected_chunks:
            return hits, 0
        kept = [h for h in hits if not looks_like_injection(h.chunk.text)]
        return kept, len(hits) - len(kept)

    def passes_gate(self, top_score: float | None) -> bool:
        return self.min_confidence is None or (top_score is not None and top_score >= self.min_confidence)

    def grounded(self, answer: str, context: str) -> bool:
        return self.grounding_threshold is None or support_score(answer, context) >= self.grounding_threshold
