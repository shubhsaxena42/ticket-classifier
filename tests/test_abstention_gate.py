"""Unit tests for the abstention gate component."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generation.abstention_gate import ABSTAIN_RESPONSE_TEXT, AbstentionGate  # type: ignore[import-not-found]


def test_abstains_when_all_scores_low() -> None:
    chunks = [{"rerank_score": 0.1}, {"rerank_score": 0.2}, {"rerank_score": 0.3}]
    gate = AbstentionGate(threshold=0.35)
    result = gate.check(chunks)
    assert result.should_abstain is True
    assert result.max_score == 0.3


def test_passes_when_one_score_high() -> None:
    chunks = [{"rerank_score": 0.2}, {"rerank_score": 0.5}, {"rerank_score": 0.1}]
    gate = AbstentionGate(threshold=0.35)
    result = gate.check(chunks)
    assert result.should_abstain is False
    assert result.max_score == 0.5


def test_abstains_on_empty_chunks() -> None:
    gate = AbstentionGate()
    result = gate.check([])
    assert result.should_abstain is True
    assert result.max_score == 0.0


def test_langgraph_returns_correct_keys_when_abstaining() -> None:
    gate = AbstentionGate(threshold=0.35)
    chunks = [{"rerank_score": 0.1, "source": "faq.json", "text": "x", "chunk_id": 0}] * 3
    state = {"reranked_chunks": chunks}
    output = gate.check_for_langgraph(state)
    assert "abstain_flag" in output
    assert "draft_response" in output
    assert output["abstain_flag"] is True
    assert output["draft_response"] == ABSTAIN_RESPONSE_TEXT
