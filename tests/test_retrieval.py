from raglab.cache import KVCache
from raglab.chunking import chunk_corpus
from raglab.retrievers import BM25Retriever, HybridRetriever


class FixedRetriever:
    def __init__(self, name, order):
        self.name, self.order = name, order

    def fit(self, chunks):
        return self

    def search_many(self, queries, k):
        return [[(i, 1.0 / (r + 1)) for r, i in enumerate(self.order[:k])] for _ in queries]


def test_bm25_finds_the_right_chunk(docs):
    chunks = chunk_corpus(docs, {"type": "paragraph"})
    hits = BM25Retriever().fit(chunks).search_many(["Who was the first ruler of Normandy?", "share of rainforest in Brazil"], 3)
    assert "Rollo" in chunks[hits[0][0][0]].text
    assert "Brazil" in chunks[hits[1][0][0]].text
    assert all(a[1] >= b[1] for a, b in zip(hits[0], hits[0][1:]))


def test_bm25_unknown_terms_score_zero(docs):
    chunks = chunk_corpus(docs, {"type": "paragraph"})
    hits = BM25Retriever().fit(chunks).search_many(["zzzz qqqq"], 2)[0]
    assert all(score == 0 for _, score in hits)


def test_rrf_rewards_agreement():
    hybrid = HybridRetriever([FixedRetriever("a", [0, 1, 2]), FixedRetriever("b", [1, 2, 0])], rrf_k=60, pool=3)
    top = hybrid.search_many(["q"], 3)[0]
    assert top[0][0] == 1  # ranked 2nd and 1st beats 1st and 3rd
    assert hybrid.name == "hybrid(a+b)"


def test_cache_roundtrip(tmp_path):
    import numpy as np

    cache = KVCache(tmp_path / "c.sqlite")
    cache.put_vectors("ns", {"k": np.array([1.0, 2.0])})
    assert cache.get_vectors("ns", ["k", "missing"])["k"].tolist() == [1.0, 2.0]
    assert (cache.hits, cache.misses) == (1, 1)
    cache.put_json("j", "x", {"a": 1})
    assert cache.get_json("j", "x") == {"a": 1}
