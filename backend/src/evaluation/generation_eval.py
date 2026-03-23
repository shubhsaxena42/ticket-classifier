"""RAGAS-based generation quality evaluator.

This module wraps RAGAS metrics (faithfulness, answer relevancy, context
precision) for post-inference evaluation. RAGAS may internally call an LLM.
By default this often expects OPENAI_API_KEY; configure a custom LLM wrapper
for your chosen provider if your environment does not use OpenAI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from evaluation.cost_tracker import CostTracker

try:
    from datasets import Dataset  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    Dataset = None  # type: ignore[assignment]
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None

try:
    from ragas import evaluate as ragas_evaluate  # type: ignore[import-not-found]
    from ragas.metrics import answer_relevancy, context_precision, faithfulness  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    ragas_evaluate = None  # type: ignore[assignment]
    answer_relevancy = None  # type: ignore[assignment]
    context_precision = None  # type: ignore[assignment]
    faithfulness = None  # type: ignore[assignment]
    _RAGAS_IMPORT_ERROR = exc
else:
    _RAGAS_IMPORT_ERROR = None


class GenerationEvaluator:
    """Evaluate generated responses using RAGAS metrics.

    Args:
        cost_tracker: Optional cost tracker for future LLM-backed metric logging.

    Returns:
        None. Use evaluate() and compare() to compute metrics.
    """

    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        if ragas_evaluate is None or faithfulness is None or answer_relevancy is None or context_precision is None:
            raise ImportError("ragas is required to instantiate GenerationEvaluator.") from _RAGAS_IMPORT_ERROR
        if Dataset is None:
            raise ImportError("datasets is required to instantiate GenerationEvaluator.") from _DATASETS_IMPORT_ERROR

        self.metrics = [faithfulness, answer_relevancy, context_precision]
        self.cost_tracker = cost_tracker

    def build_ragas_dataset(
        self,
        tickets: List[str],
        responses: List[str],
        contexts: List[List[str]],
        abstained: List[bool],
    ) -> Any:
        """Build a Hugging Face Dataset in RAGAS format, excluding abstentions.

        Args:
            tickets: Cleaned ticket texts (questions).
            responses: Generated response texts.
            contexts: Per-ticket context text lists.
            abstained: Per-ticket abstention flags.

        Returns:
            datasets.Dataset ready for ragas.evaluate.
        """

        rows = [
            {"question": ticket, "answer": response, "contexts": context}
            for ticket, response, context, abstain in zip(tickets, responses, contexts, abstained)
            if (not abstain) and str(response).strip()
        ]
        if Dataset is None:
            raise ImportError("datasets is required to build RAGAS datasets.") from _DATASETS_IMPORT_ERROR
        return Dataset.from_list(rows)

    def evaluate(
        self,
        tickets: List[str],
        responses: List[str],
        contexts: List[List[str]],
        abstained: List[bool],
    ) -> Dict[str, Any]:
        """Run RAGAS evaluation and return aggregate metric scores.

        Args:
            tickets: Cleaned ticket texts.
            responses: Generated responses.
            contexts: Per-ticket source texts.
            abstained: Per-ticket abstention flags.

        Returns:
            Metric dictionary with score values and counts.
        """

        dataset = self.build_ragas_dataset(
            tickets=tickets,
            responses=responses,
            contexts=contexts,
            abstained=abstained,
        )
        if len(dataset) == 0:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "evaluated_count": 0,
                "abstained_count": int(sum(abstained)),
            }

        if ragas_evaluate is None:
            raise ImportError("ragas is required to run generation evaluation.") from _RAGAS_IMPORT_ERROR
        result = ragas_evaluate(dataset, metrics=self.metrics)
        scores = result.to_pandas().mean().to_dict()
        return {
            "faithfulness": float(scores.get("faithfulness", 0.0)),
            "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
            "context_precision": float(scores.get("context_precision", 0.0)),
            "evaluated_count": int(len(dataset)),
            "abstained_count": int(sum(abstained)),
        }

    def compare(self, baseline_kwargs: Dict[str, Any], improved_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Compare RAGAS metrics between baseline and improved pipelines.

        Args:
            baseline_kwargs: Keyword args for evaluate() for baseline pipeline.
            improved_kwargs: Keyword args for evaluate() for improved pipeline.

        Returns:
            Dictionary with baseline, improved, and metric deltas.
        """

        baseline = self.evaluate(**baseline_kwargs)
        improved = self.evaluate(**improved_kwargs)

        tracked_keys = ["faithfulness", "answer_relevancy", "context_precision"]
        return {
            "baseline": baseline,
            "improved": improved,
            "delta": {
                key: round(float(improved.get(key, 0.0)) - float(baseline.get(key, 0.0)), 4)
                for key in tracked_keys
            },
        }
