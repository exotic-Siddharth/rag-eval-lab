"""Corpus and labelled-question loading.

Gold labels are character spans in the source documents, not chunk ids, so the same question set
scores any chunking strategy fairly: a chunk is relevant iff it fully contains a gold answer span.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .text import normalize_answer

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class Question:
    qid: str
    question: str
    doc_id: str | None
    answers: list[str]
    spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.answers)


def download_squad(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_DEV_URL, path)
    return path


def load_squad(path: str | Path) -> tuple[list[Document], list[Question]]:
    """SQuAD 2.0 -> one document per Wikipedia article (paragraphs joined by blank lines).

    Rebuilding whole articles is what makes chunking a real design choice; answer offsets are
    shifted into article coordinates so they stay exact.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    docs, questions = [], []
    for a_idx, article in enumerate(raw):
        doc_id = f"d{a_idx:03d}"
        title = article["title"].replace("_", " ")
        parts, offset = [], 0
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                answers, spans = [], []
                if not qa.get("is_impossible", False):
                    for ans in qa["answers"]:
                        s = offset + ans["answer_start"]
                        spans.append((s, s + len(ans["text"])))
                        answers.append(ans["text"])
                questions.append(Question(qa["id"], qa["question"].strip(), doc_id, answers, sorted(set(spans))))
            parts.append(ctx)
            offset += len(ctx) + 2
        docs.append(Document(doc_id, title, "\n\n".join(parts)))
    return docs, questions


def load_corpus_dir(path: str | Path) -> list[Document]:
    """Any folder of .txt/.md files becomes a corpus (one document per file)."""
    files = sorted(p for p in Path(path).rglob("*") if p.suffix.lower() in {".txt", ".md"})
    return [Document(f"f{i:04d}", p.stem, p.read_text(encoding="utf-8")) for i, p in enumerate(files)]


def load_questions_jsonl(path: str | Path) -> list[Question]:
    """One JSON object per line: {"question": ..., "answers": [...]}; empty answers = unanswerable."""
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            row = json.loads(line)
            out.append(Question(row.get("id", f"q{i}"), row["question"], row.get("doc_id"), row.get("answers", [])))
    return out


def load_corpus(spec: dict) -> tuple[list[Document], list[Question]]:
    kind = spec.get("type", "squad")
    if kind == "squad":
        return load_squad(download_squad(spec.get("path", "data/dev-v2.0.json")))
    if kind == "dir":
        questions = load_questions_jsonl(spec["questions"]) if spec.get("questions") else []
        return load_corpus_dir(spec["path"]), questions
    raise ValueError(f"unknown corpus type: {kind}")


def split_by_document(questions: list[Question], tune_frac: float = 0.3, seed: int = 0):
    """Split questions into tune/test by *document*, so thresholds are never tuned on test topics."""

    def bucket(doc_id: str | None) -> float:
        h = hashlib.sha1(f"{seed}:{doc_id}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    tune = [q for q in questions if bucket(q.doc_id) < tune_frac]
    test = [q for q in questions if bucket(q.doc_id) >= tune_frac]
    return tune, test


def sample(questions: list[Question], n_answerable: int | None, n_unanswerable: int | None = 0, seed: int = 0):
    rng = random.Random(seed)
    ans = [q for q in questions if q.answerable]
    una = [q for q in questions if not q.answerable]
    rng.shuffle(ans)
    rng.shuffle(una)
    ans = ans if n_answerable is None else ans[:n_answerable]
    una = una if n_unanswerable is None else una[:n_unanswerable]
    return ans + una


def is_relevant(chunk, q: Question) -> bool:
    if q.spans:
        return chunk.doc_id == q.doc_id and any(chunk.start <= s and e <= chunk.end for s, e in q.spans)
    text = normalize_answer(chunk.text)
    return any(normalize_answer(a) and normalize_answer(a) in text for a in q.answers)
