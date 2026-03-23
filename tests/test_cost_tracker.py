"""Unit tests for token cost tracking and summary aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.cost_tracker import CostTracker  # type: ignore[import-not-found]


def test_cost_calculation_correct() -> None:
    tracker = CostTracker()
    tracker.log("generation", input_tokens=1000, output_tokens=200)
    summary = tracker.summary()
    expected_cost = (1000 / 1000 * 0.00025) + (200 / 1000 * 0.00125)
    assert abs(summary["total_cost_usd"] - expected_cost) < 0.000001


def test_summary_empty_tracker_returns_zeros() -> None:
    tracker = CostTracker()
    summary = tracker.summary()
    assert summary["total_cost_usd"] == 0.0
    assert summary["total_calls"] == 0


def test_by_component_grouping() -> None:
    tracker = CostTracker()
    tracker.log("generation", 100, 50)
    tracker.log("hyde", 200, 80)
    tracker.log("generation", 150, 60)
    summary = tracker.summary()
    assert "generation" in summary["by_component"]
    assert summary["by_component"]["generation"]["call_count"] == 2
