"""Unit tests for generation schemas and state mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest  # type: ignore[import-not-found]
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generation.schemas import GenerationOutput, TicketPrediction  # type: ignore[import-not-found]


def test_generation_output_rejects_empty_citations() -> None:
    with pytest.raises(ValidationError):
        GenerationOutput(draft_response="answer", citations=[])


def test_generation_output_rejects_invalid_citation_index() -> None:
    with pytest.raises(ValidationError):
        GenerationOutput(draft_response="answer", citations=[3])


def test_ticket_prediction_from_state() -> None:
    state = {
        "predicted_category": "technical",
        "predicted_priority": "high",
        "reranked_chunks": [
            {"source": "faq.json", "chunk_id": 0, "text": "x", "rerank_score": 0.8},
            {"source": "policy.json", "chunk_id": 1, "text": "y", "rerank_score": 0.7},
            {"source": "kb.json", "chunk_id": 2, "text": "z", "rerank_score": 0.6},
        ],
        "draft_response": "Reset your password from account settings.",
        "confidence_score": 0.91,
        "abstain_flag": False,
        "citations": [0, 2],
        "tier_used": "setfit",
        "routing_action": "auto_route",
        "retrieval_max_score": 0.8,
    }

    pred = TicketPrediction.from_state(state=state, ticket_id="ticket_001")

    assert pred.ticket_id == "ticket_001"
    assert pred.top_3_retrieved_sources == "faq.json|policy.json|kb.json"
    assert pred.citations == "0|2"
    assert pred.predicted_category == "technical"
