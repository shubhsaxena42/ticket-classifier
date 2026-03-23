from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class TicketPrediction(BaseModel):
    predicted_category: str
    predicted_priority: str
    confidence_score: Optional[float] = None
    tier_used: Optional[str] = None
    abstain_flag: Optional[bool] = None
    routing_action: Literal[
        "auto_route",
        "suggest",
        "manual_review",
        "escalate_to_tier2",
        "escalate_to_tier3",
    ]


class FeedbackEvent(BaseModel):
    ticket_id: str
    predicted_category: str
    predicted_priority: str
    agent_correction: Optional[str] = None
    response_rating: Optional[int] = None
    model_version: str
    timestamp: datetime
