"""Pydantic schemas for generation outputs, predictions, and feedback events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class GenerationOutput(BaseModel):
    """Structured generation response with grounded citations.

    Args:
        draft_response: Grounded support response text.
        citations: Source indices used to build the response.

    Returns:
        Validated generation payload for downstream state updates.
    """

    draft_response: str = Field(
        description=(
            "Concise response to the support ticket. "
            "Must be derived entirely from the provided sources. "
            "Do not include information not present in the sources."
        )
    )
    citations: List[int] = Field(
        description=(
            "List of source indices (0, 1, or 2) used to construct the response. "
            "Must contain at least one index when draft_response is non-empty. "
            "If no source is relevant, leave draft_response empty and citations empty."
        )
    )

    @field_validator("citations")
    @classmethod
    def validate_citation_range(cls, value: List[int]) -> List[int]:
        """Enforce that citations only reference source indices 0, 1, or 2."""

        for citation in value:
            if citation not in (0, 1, 2):
                raise ValueError("citations values must be in range [0, 1, 2]")
        return value

    @field_validator("draft_response")
    @classmethod
    def normalize_draft_response(cls, value: str) -> str:
        """Normalize whitespace around draft response content."""

        return value.strip()

    @model_validator(mode="after")
    def validate_response_citation_consistency(self) -> "GenerationOutput":
        """Require non-empty citations for non-empty responses and vice versa."""

        if self.citations and not self.draft_response:
            raise ValueError("draft_response must not be empty when citations are provided")
        if not self.citations and self.draft_response:
            raise ValueError("citations must not be empty when draft_response is provided")
        return self


class TicketPrediction(BaseModel):
    """Final ticket prediction row for predictions.csv output."""

    ticket_id: str
    predicted_category: str
    predicted_priority: str
    top_3_retrieved_sources: str
    draft_response: str
    confidence_score: float
    abstain_flag: bool
    citations: str
    tier_used: str
    routing_action: str
    retrieval_max_score: float

    @classmethod
    def from_state(cls, state: Dict[str, Any], ticket_id: str) -> "TicketPrediction":
        """Build a TicketPrediction from a completed TicketState-like dictionary.

        Args:
            state: Final pipeline state containing classification and generation fields.
            ticket_id: Unique ticket identifier.

        Returns:
            TicketPrediction instance ready for serialization.
        """

        reranked_chunks = state.get("reranked_chunks", []) or []
        citations = state.get("citations", []) or []

        top_3_retrieved_sources = "|".join(str(chunk.get("source", "")) for chunk in reranked_chunks)
        citations_str = "|".join(str(i) for i in citations)

        return cls(
            ticket_id=str(ticket_id),
            predicted_category=str(state.get("predicted_category", "")),
            predicted_priority=str(state.get("predicted_priority", "")),
            top_3_retrieved_sources=top_3_retrieved_sources,
            draft_response=str(state.get("draft_response", "")),
            confidence_score=float(state.get("confidence_score", 0.0)),
            abstain_flag=bool(state.get("abstain_flag", False)),
            citations=citations_str,
            tier_used=str(state.get("tier_used", "")),
            routing_action=str(state.get("routing_action", "")),
            retrieval_max_score=float(state.get("retrieval_max_score", 0.0)),
        )


class FeedbackEvent(BaseModel):
    """Human feedback event for post-hoc quality and correction tracking."""

    ticket_id: str
    predicted_category: str
    predicted_priority: str
    agent_correction: Optional[str] = None
    response_rating: Optional[int] = None
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
