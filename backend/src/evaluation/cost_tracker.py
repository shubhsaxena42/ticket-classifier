"""Token usage and cost accounting utilities for model-backed components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

HAIKU_INPUT_COST_PER_1K = 0.00025
HAIKU_OUTPUT_COST_PER_1K = 0.00125

PathLike = Union[str, Path]


class CostTracker:
    """Track per-call token usage and aggregate dollar cost.

    Args:
        None.

    Returns:
        None. Use log(), summary(), save(), and per_ticket_cost().
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def log(
        self,
        component: str,
        input_tokens: int,
        output_tokens: int,
        ticket_id: Optional[str] = None,
    ) -> None:
        """Append one usage record and computed cost.

        Args:
            component: Logical component name (e.g., generation).
            input_tokens: Prompt/input token count.
            output_tokens: Response/output token count.
            ticket_id: Optional ticket identifier for per-ticket breakdown.

        Returns:
            None.
        """

        input_cost = float(input_tokens) / 1000.0 * HAIKU_INPUT_COST_PER_1K
        output_cost = float(output_tokens) / 1000.0 * HAIKU_OUTPUT_COST_PER_1K
        total_cost = input_cost + output_cost

        self.records.append(
            {
                "component": str(component),
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "cost_usd": float(total_cost),
                "ticket_id": str(ticket_id) if ticket_id is not None else None,
            }
        )

    def summary(self) -> Dict[str, Any]:
        """Summarize usage and cost by component with overall totals.

        Args:
            None.

        Returns:
            Dictionary with by_component and overall totals.
        """

        by_component: Dict[str, Dict[str, Any]] = {}

        for record in self.records:
            component = str(record["component"])
            component_agg = by_component.setdefault(
                component,
                {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost_usd": 0.0,
                    "call_count": 0,
                },
            )
            component_agg["total_input_tokens"] += int(record["input_tokens"])
            component_agg["total_output_tokens"] += int(record["output_tokens"])
            component_agg["total_cost_usd"] += float(record["cost_usd"])
            component_agg["call_count"] += 1

        total_input_tokens = sum(int(record["input_tokens"]) for record in self.records)
        total_output_tokens = sum(int(record["output_tokens"]) for record in self.records)
        total_cost_usd = sum(float(record["cost_usd"]) for record in self.records)
        total_calls = len(self.records)

        return {
            "by_component": by_component,
            "total_cost_usd": float(total_cost_usd),
            "total_input_tokens": int(total_input_tokens),
            "total_output_tokens": int(total_output_tokens),
            "total_calls": int(total_calls),
        }

    def save(self, path: PathLike) -> None:
        """Save aggregate summary and per-ticket breakdown to JSON.

        Args:
            path: Output JSON path.

        Returns:
            None.
        """

        output_path = Path(path)
        summary = self.summary()

        by_ticket: Dict[str, Dict[str, Any]] = {}
        for record in self.records:
            ticket_id = record.get("ticket_id")
            if not ticket_id:
                continue
            ticket_agg = by_ticket.setdefault(
                str(ticket_id),
                {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost_usd": 0.0,
                    "call_count": 0,
                },
            )
            ticket_agg["total_input_tokens"] += int(record["input_tokens"])
            ticket_agg["total_output_tokens"] += int(record["output_tokens"])
            ticket_agg["total_cost_usd"] += float(record["cost_usd"])
            ticket_agg["call_count"] += 1

        payload = {
            **summary,
            "per_ticket": by_ticket,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def per_ticket_cost(self) -> float:
        """Return average cost per logged call.

        Args:
            None.

        Returns:
            Average cost in USD across all logged calls.
        """

        summary = self.summary()
        return float(summary["total_cost_usd"]) / max(int(summary["total_calls"]), 1)
