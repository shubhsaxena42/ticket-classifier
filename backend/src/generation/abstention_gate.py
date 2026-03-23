"""Abstention gate for retrieval-grounding checks before response generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

ABSTAIN_RESPONSE_TEXT = (
    "Insufficient information in the knowledge base to answer this "
    "question. Please escalate to a human agent."
)


@dataclass
class GateResult:
    """Result container for abstention decisions.

    Args:
        should_abstain: Whether generation should be skipped.
        max_score: Maximum rerank score observed across retrieved chunks.
        reason: Optional reason string for diagnostics.
    """

    should_abstain: bool
    max_score: float
    reason: Optional[str]


class AbstentionGate:
    """Decide whether retrieval evidence is strong enough for generation.

    Args:
        threshold: Minimum acceptable max rerank score.

    Returns:
        None. Use check/check_for_langgraph for gate outcomes.
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = float(threshold)

    def check(self, reranked_chunks: List[Dict[str, Any]]) -> GateResult:
        """Evaluate abstention decision from reranked retrieval chunks.

        Args:
            reranked_chunks: Retrieved chunks with rerank_score fields.

        Returns:
            GateResult describing abstention, max score, and reason.
        """

        if not reranked_chunks:
            return GateResult(
                should_abstain=True,
                max_score=0.0,
                reason="no_chunks_retrieved",
            )

        max_score = max(float(chunk.get("rerank_score", 0.0)) for chunk in reranked_chunks)
        if max_score < self.threshold:
            return GateResult(
                should_abstain=True,
                max_score=max_score,
                reason="max_rerank_score_below_threshold",
            )

        return GateResult(
            should_abstain=False,
            max_score=max_score,
            reason=None,
        )

    def check_for_langgraph(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node adapter for abstention output state updates.

        Args:
            state: Pipeline state containing reranked_chunks.

        Returns:
            Partial state update for abstention outcomes and max score.
        """

        reranked_chunks = state.get("reranked_chunks", [])
        if not isinstance(reranked_chunks, list):
            reranked_chunks = []

        result = self.check(reranked_chunks=reranked_chunks)
        if result.should_abstain:
            return {
                "abstain_flag": True,
                "draft_response": ABSTAIN_RESPONSE_TEXT,
                "citations": [],
                "top_3_sources": [str(chunk.get("source", "")) for chunk in reranked_chunks],
                "retrieval_max_score": float(result.max_score),
            }

        return {
            "abstain_flag": False,
            "retrieval_max_score": float(result.max_score),
        }
