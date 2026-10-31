"""Chunking strategies. Every chunk keeps exact character offsets into its source document."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .data import Document
from .text import sentence_spans


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start: int
    end: int


def _make(doc: Document, start: int, end: int) -> Chunk:
    return Chunk(f"{doc.doc_id}:{start}-{end}", doc.doc_id, doc.title, doc.text[start:end], start, end)


def chunk_fixed(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Sliding window of `size` words with `overlap` words shared between neighbours."""
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    words = [m.span() for m in re.finditer(r"\S+", doc.text)]
    chunks, step = [], size - overlap
    for i in range(0, max(len(words) - overlap, 1), step):
        window = words[i : i + size]
        if window:
            chunks.append(_make(doc, window[0][0], window[-1][1]))
    return chunks


def chunk_sentences(
    doc: Document, max_words: int, overlap_sentences: int = 0, start: int = 0, end: int | None = None
) -> list[Chunk]:
    """Pack whole sentences up to `max_words`; never cuts a sentence (or an answer) in half."""
    sents = sentence_spans(doc.text, start, end)
    lengths = [len(doc.text[s:e].split()) for s, e in sents]
    chunks, i = [], 0
    while i < len(sents):
        j, words = i, 0
        while j < len(sents) and (j == i or words + lengths[j] <= max_words):
            words += lengths[j]
            j += 1
        chunks.append(_make(doc, sents[i][0], sents[j - 1][1]))
        if j >= len(sents):
            break
        i = max(j - overlap_sentences, i + 1)
    return chunks


def chunk_paragraphs(doc: Document, max_words: int = 400) -> list[Chunk]:
    """Native paragraphs; over-long ones fall back to sentence packing."""
    chunks = []
    for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", doc.text):
        if len(m.group().split()) <= max_words:
            chunks.append(_make(doc, m.start(), m.end()))
        else:
            chunks.extend(chunk_sentences(doc, max_words, start=m.start(), end=m.end()))
    return chunks


def chunker_name(spec: dict) -> str:
    kind = spec["type"]
    if kind == "fixed":
        return f"fixed-{spec['size']}w/{spec['overlap']}"
    if kind == "sentence":
        return f"sentence-{spec['max_words']}w" + (f"+{spec['overlap_sentences']}s" if spec.get("overlap_sentences") else "")
    if kind == "paragraph":
        return "paragraph"
    raise ValueError(f"unknown chunker: {kind}")


def chunk_corpus(docs: list[Document], spec: dict) -> list[Chunk]:
    kind = spec["type"]
    out: list[Chunk] = []
    for doc in docs:
        if kind == "fixed":
            out.extend(chunk_fixed(doc, spec["size"], spec["overlap"]))
        elif kind == "sentence":
            out.extend(chunk_sentences(doc, spec["max_words"], spec.get("overlap_sentences", 0)))
        elif kind == "paragraph":
            out.extend(chunk_paragraphs(doc, spec.get("max_words", 400)))
        else:
            raise ValueError(f"unknown chunker: {kind}")
    return out
