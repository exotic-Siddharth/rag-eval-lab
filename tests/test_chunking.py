import pytest

from raglab.chunking import chunk_corpus, chunk_fixed, chunk_paragraphs, chunk_sentences, chunker_name
from raglab.data import Document, Question, is_relevant
from raglab.text import sentence_spans


@pytest.mark.parametrize("spec", [
    {"type": "fixed", "size": 8, "overlap": 2},
    {"type": "sentence", "max_words": 12, "overlap_sentences": 1},
    {"type": "paragraph"},
])
def test_offsets_are_exact(docs, spec):
    for c in chunk_corpus(docs, spec):
        doc = next(d for d in docs if d.doc_id == c.doc_id)
        assert doc.text[c.start:c.end] == c.text
        assert c.chunk_id == f"{c.doc_id}:{c.start}-{c.end}"


def test_fixed_window_overlap_and_full_coverage():
    doc = Document("x", "t", " ".join(f"w{i}" for i in range(20)))
    chunks = chunk_fixed(doc, size=8, overlap=3)
    assert [len(c.text.split()) for c in chunks][:-1] == [8] * (len(chunks) - 1)
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks[-1].text.split()[-1] == "w19"


def test_fixed_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_fixed(Document("x", "t", "a b c"), size=4, overlap=4)


def test_sentence_chunks_never_split_sentences(docs):
    doc = docs[0]
    bounds = {s for s, _ in sentence_spans(doc.text)} | {e for _, e in sentence_spans(doc.text)}
    for c in chunk_sentences(doc, max_words=10):
        assert c.start in bounds and c.end in bounds


def test_paragraph_chunks_follow_blank_lines(docs):
    assert [c.text.split()[0] for c in chunk_paragraphs(docs[1])] == ["The", "Brazil"]


def test_relevance_uses_char_spans(docs):
    doc = docs[0]
    s = doc.text.index("Rollo")
    q = Question("q", "Who was the first ruler?", "d0", ["Rollo"], [(s, s + 5)])
    chunks = chunk_corpus([doc], {"type": "paragraph"})
    assert [is_relevant(c, q) for c in chunks] == [False, True]


def test_chunker_names():
    assert chunker_name({"type": "fixed", "size": 128, "overlap": 32}) == "fixed-128w/32"
    assert chunker_name({"type": "sentence", "max_words": 120}) == "sentence-120w"
