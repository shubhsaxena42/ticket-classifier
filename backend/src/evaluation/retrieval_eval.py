"""Retrieval quality evaluation utilities using Groq as a relevance judge."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import Groq
import numpy as np

from evaluation.cost_tracker import CostTracker


def _default_env_path() -> Path:
    """Return backend .env path used for local API-key fallback loading."""

    return Path(__file__).resolve().parents[2] / ".env"


def _read_key_from_env_file(env_path: Optional[Path], key_name: str) -> Optional[str]:
    """Read one key from a simple KEY=VALUE env file.

    Args:
        env_path: Optional custom env file path.
        key_name: Variable name to resolve.

    Returns:
        The resolved string value or None when not found.
    """

    path = env_path if env_path is not None else _default_env_path()
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key_name:
            return v.strip().strip('"').strip("'")
    return None


class RetrievalEvaluator:
    """Evaluate retrieval quality with MRR@3 and NDCG@3.

    This evaluator uses Groq as a relevance judge for top retrieved chunks.

    Args:
        model: Groq model id for relevance judging.
        cost_tracker: Optional tracker for API token/cost logging.
        groq_api_key: Optional explicit API key. Falls back to GROQ_API_KEY.

    Returns:
        None. Use evaluate() and compare() for metrics.
    """

    def __init__(
        self,
        model: str = "qwen/qwen3-32b",
        cost_tracker: Optional[CostTracker] = None,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        key = (groq_api_key or gemini_api_key or os.environ.get("GROQ_API_KEY", "") or "").strip()
        if not key:
            file_key = _read_key_from_env_file(env_path=None, key_name="GROQ_API_KEY")
            key = file_key.strip() if file_key else ""
        if not key:
            raise RuntimeError("GROQ_API_KEY is required for RetrievalEvaluator.")

        self.client = Groq(api_key=key)
        self.model = model
        self.cost_tracker = cost_tracker

    def judge_relevance(
        self,
        ticket: str,
        chunk: str,
        cost_tracker: Optional[CostTracker] = None,
    ) -> bool:
        """Judge whether a chunk helps answer a ticket.

        Args:
            ticket: Cleaned ticket text.
            chunk: Retrieved chunk text.
            cost_tracker: Optional per-call tracker override.

        Returns:
            True if judged relevant, otherwise False.
        """

        prompt = (
            f"Ticket: {ticket}\n\n"
            f"Passage: {chunk}\n\n"
            "Does this passage contain information that directly helps "
            "answer the ticket? Answer with exactly one word: "
            "relevant or irrelevant."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )

        tracker = cost_tracker or self.cost_tracker
        usage_meta = getattr(response, "usage", None)
        if tracker is not None and usage_meta is not None:
            input_tokens = int(getattr(usage_meta, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage_meta, "completion_tokens", 0) or 0)
            tracker.log("retrieval_judge", input_tokens=input_tokens, output_tokens=output_tokens)

        text = str(response.choices[0].message.content or "").strip().lower()
        first_word = text.split()[0].strip(".,!?:;\"'()[]{}") if text else ""
        return first_word == "relevant"

    def compute_mrr(self, relevance_lists: List[List[bool]]) -> float:
        """Compute MRR@3 from binary relevance lists.

        Args:
            relevance_lists: Per-ticket boolean relevance at ranks 1..3.

        Returns:
            Mean reciprocal rank at 3.
        """

        reciprocal_ranks: List[float] = []
        for rel_list in relevance_lists:
            rr = 0.0
            for rank, is_rel in enumerate(rel_list[:3], start=1):
                if is_rel:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    def compute_ndcg(self, relevance_lists: List[List[bool]]) -> float:
        """Compute NDCG@3 from binary relevance lists.

        Args:
            relevance_lists: Per-ticket boolean relevance at ranks 1..3.

        Returns:
            Mean NDCG at 3.
        """

        def dcg(rel_list: List[bool]) -> float:
            return float(
                sum((1.0 if rel else 0.0) / np.log2(rank + 1) for rank, rel in enumerate(rel_list[:3], start=1))
            )

        def ideal_dcg(rel_list: List[bool]) -> float:
            sorted_rel = sorted(rel_list[:3], reverse=True)
            return dcg(sorted_rel)

        ndcg_scores: List[float] = []
        for rel_list in relevance_lists:
            idcg = ideal_dcg(rel_list)
            ndcg_scores.append(dcg(rel_list) / idcg if idcg > 0 else 0.0)
        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    def evaluate(self, tickets: List[str], retrieved_sets: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Evaluate one retrieval pipeline with MRR@3 and NDCG@3.

        Args:
            tickets: Cleaned ticket texts.
            retrieved_sets: Top-3 chunk sets, one list per ticket.

        Returns:
            Dictionary with mrr@3 and ndcg@3.
        """

        relevance_lists: List[List[bool]] = []
        for ticket, chunks in zip(tickets, retrieved_sets):
            rel_list = [self.judge_relevance(ticket=ticket, chunk=str(chunk.get("text", ""))) for chunk in chunks]
            relevance_lists.append(rel_list)

        return {
            "mrr@3": self.compute_mrr(relevance_lists),
            "ndcg@3": self.compute_ndcg(relevance_lists),
        }

    def compare(
        self,
        tickets: List[str],
        baseline_results: List[List[Dict[str, Any]]],
        improved_results: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Compare baseline and improved retrieval pipelines.

        Args:
            tickets: Cleaned ticket texts used for evaluation.
            baseline_results: Baseline top-3 retrieval outputs.
            improved_results: Improved top-3 retrieval outputs.

        Returns:
            Dictionary with baseline, improved, and delta metrics.
        """

        baseline_metrics = self.evaluate(tickets=tickets, retrieved_sets=baseline_results)
        improved_metrics = self.evaluate(tickets=tickets, retrieved_sets=improved_results)

        return {
            "baseline": baseline_metrics,
            "improved": improved_metrics,
            "delta": {
                key: round(float(improved_metrics[key]) - float(baseline_metrics[key]), 4)
                for key in baseline_metrics
            },
        }
