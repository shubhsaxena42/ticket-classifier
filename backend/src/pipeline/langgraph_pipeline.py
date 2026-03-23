"""LangGraph wiring for support ticket inference.

This module wires retrieval grading and grounded response generation nodes into a
StateGraph. It expects upstream nodes for scrubbing, classification, HyDE, and
reranking to be provided by the caller.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, TypedDict

from generation.abstention_gate import AbstentionGate
from generation.generator import ResponseGenerator

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    END = None  # type: ignore[assignment]
    START = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    _LANGGRAPH_IMPORT_ERROR = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None


class TicketState(TypedDict, total=False):
    """Shared state keys used by the support-ticket LangGraph pipeline."""

    cleaned_message: str
    hyde_queries: List[str]
    reranked_chunks: List[Dict[str, Any]]
    predicted_category: str
    predicted_priority: str
    confidence_score: float
    tier_used: str
    routing_action: str
    abstain_flag: bool
    draft_response: str
    citations: List[int]
    top_3_sources: List[str]
    retrieval_max_score: float


def build_support_graph(
    scrub_node: Callable[[TicketState], Dict[str, Any]],
    classify_node: Callable[[TicketState], Dict[str, Any]],
    hyde_node: Callable[[TicketState], Dict[str, Any]],
    rerank_node: Callable[[TicketState], Dict[str, Any]],
    output_node: Callable[[TicketState], Dict[str, Any]],
    abstention_gate: AbstentionGate,
    response_generator: ResponseGenerator,
) -> Any:
    """Build and compile a support ticket LangGraph StateGraph.

    Args:
        scrub_node: Node function that prepares cleaned ticket text.
        classify_node: Node function producing classification outputs.
        hyde_node: Node function producing HyDE query expansions.
        rerank_node: Node function producing reranked_chunks.
        output_node: Final node that formats output record(s).
        abstention_gate: AbstentionGate instance for grade_node.
        response_generator: ResponseGenerator instance for generate_node.

    Returns:
        Compiled LangGraph graph instance.

    Raises:
        ImportError: If langgraph is not installed.
    """

    if StateGraph is None or START is None or END is None:
        raise ImportError("langgraph is required to build the StateGraph pipeline.") from _LANGGRAPH_IMPORT_ERROR

    graph = StateGraph(TicketState)

    graph.add_node("scrub_node", scrub_node)
    graph.add_node("classify_node", classify_node)
    graph.add_node("hyde_node", hyde_node)
    graph.add_node("rerank_node", rerank_node)
    graph.add_node("grade_node", abstention_gate.check_for_langgraph)
    graph.add_node("generate_node", response_generator.generate_for_langgraph)
    graph.add_node("output_node", output_node)

    graph.add_edge(START, "scrub_node")
    graph.add_edge("scrub_node", "classify_node")
    graph.add_edge("scrub_node", "hyde_node")
    graph.add_edge("hyde_node", "rerank_node")
    graph.add_edge("rerank_node", "grade_node")

    graph.add_conditional_edges(
        "grade_node",
        lambda state: "generate_node" if not bool(state.get("abstain_flag", False)) else "output_node",
        {
            "generate_node": "generate_node",
            "output_node": "output_node",
        },
    )

    graph.add_edge("generate_node", "output_node")
    graph.add_edge("output_node", END)

    return graph.compile()
