import math

import pytest

from raglab import metrics as M


def test_recall_mrr_ndcg():
    rels = [[False, True, False], [False, False, False], [True, False, True]]
    assert M.recall_at_k(rels, 1) == pytest.approx(1 / 3)
    assert M.recall_at_k(rels, 3) == pytest.approx(2 / 3)
    assert M.mrr_at_k(rels, 3) == pytest.approx((0.5 + 0 + 1) / 3)
    # second question has no relevant chunk at all -> contributes 0
    ndcg = M.ndcg_at_k(rels, [1, 0, 2], 3)
    expected_q3 = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg == pytest.approx((1 / math.log2(3) + 0 + expected_q3) / 3)


def test_squad_em_f1():
    assert M.exact_match("The Normans", ["Normans"]) == 1.0
    assert M.f1_score("Norse raiders from Denmark", ["Norse raiders"]) == pytest.approx(2 * 0.5 * 1 / 1.5)
    # unanswerable: abstaining (empty prediction) is the correct answer
    assert M.f1_score("", []) == 1.0
    assert M.f1_score("Berlin", []) == 0.0
    assert M.f1_score("", ["Rollo"]) == 0.0


def test_auroc():
    assert M.auroc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert M.auroc([0.1, 0.2, 0.9, 0.8], [True, True, False, False]) == 0.0
    assert M.auroc([0.5, 0.5], [True, False]) == 0.5


def test_threshold_for_pass_rate():
    scores = list(range(100))
    tau = M.threshold_for_pass_rate(scores, 0.9)
    assert sum(s >= tau for s in scores) == 90
