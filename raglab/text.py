"""Tokenisation, normalisation and sentence splitting shared by every component."""

from __future__ import annotations

import re
import string

_WORD = re.compile(r"\w+", re.UNICODE)
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")

STOPWORDS = frozenset(
    """a an and are as at be been but by can did do does for from had has have he her his how i if in
    into is it its many much not of on or she so than that the their them there these they this those
    to was were what when where which while who whom whose why will with would you""".split()
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Character spans of sentences in text[start:end]; blank lines always end a sentence."""
    end = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    for block in re.finditer(r"[^\n]+", text[start:end]):
        b0 = start + block.start()
        block_text = block.group()
        cur = 0
        for m in _SENT_END.finditer(block_text):
            spans.append((b0 + cur, b0 + m.start() + len(m.group().rstrip())))
            cur = m.end()
        if block_text[cur:].strip():
            spans.append((b0 + cur, b0 + len(block_text.rstrip())))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def normalize_answer(s: str) -> str:
    """Official SQuAD normalisation: lower case, drop punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())
