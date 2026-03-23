"""Unit tests for retrieval metric computations."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.retrieval_eval import RetrievalEvaluator  # type: ignore[import-not-found]


class _DummyEvaluator:
    def compute_mrr(self, relevance_lists):
        return RetrievalEvaluator.compute_mrr(self, relevance_lists)

    def compute_ndcg(self, relevance_lists):
        return RetrievalEvaluator.compute_ndcg(self, relevance_lists)


def test_mrr_perfect_ranking() -> None:
    relevance_lists = [[True, False, False], [True, False, False]]
    evaluator = _DummyEvaluator()
    mrr = evaluator.compute_mrr(relevance_lists)
    assert mrr == 1.0


def test_mrr_no_relevant() -> None:
    relevance_lists = [[False, False, False]]
    evaluator = _DummyEvaluator()
    mrr = evaluator.compute_mrr(relevance_lists)
    assert mrr == 0.0


def test_mrr_relevant_at_rank_2() -> None:
    relevance_lists = [[False, True, False]]
    evaluator = _DummyEvaluator()
    mrr = evaluator.compute_mrr(relevance_lists)
    assert abs(mrr - 0.5) < 0.001


def test_ndcg_perfect_ranking() -> None:
    relevance_lists = [[True, True, True]]
    evaluator = _DummyEvaluator()
    ndcg = evaluator.compute_ndcg(relevance_lists)
    assert abs(ndcg - 1.0) < 0.001
