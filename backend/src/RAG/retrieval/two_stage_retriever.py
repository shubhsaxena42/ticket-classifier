"""Two-stage retrieval orchestrator for support ticket RAG.

This module defines TwoStageRetriever, which runs sparse+dense recall,
fuses results with RRF, and applies cross-encoder reranking.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .bm25_retriever import BM25Retriever
from .faiss_retriever import FAISSRetriever
from .reranker import CrossEncoderReranker
from .rrf import reciprocal_rank_fusion


class TwoStageRetriever:
    """Coordinate recall, fusion, and reranking for ticket retrieval.

    Parameters
    ----------
    bm25_retriever:
        Preconfigured BM25Retriever instance.
    faiss_retriever:
        Preconfigured FAISSRetriever instance.
    reranker:
        Preconfigured CrossEncoderReranker instance.
    recall_top_k:
        Number of chunks returned per retriever per query.
    rrf_k:
        RRF smoothing constant.
    rrf_top_n:
        Number of fused candidates passed to reranker.
    final_top_k:
        Number of final reranked chunks to return.

    Returns
    -------
    None
        Initializes a retrieval orchestrator instance.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        faiss_retriever: FAISSRetriever,
        reranker: CrossEncoderReranker,
        recall_top_k: int = 15,
        rrf_k: int = 60,
        rrf_top_n: int = 30,
        final_top_k: int = 3,
    ):
        self.bm25_retriever = bm25_retriever
        self.faiss_retriever = faiss_retriever
        self.reranker = reranker
        self.recall_top_k = recall_top_k
        self.rrf_k = rrf_k
        self.rrf_top_n = rrf_top_n
        self.final_top_k = final_top_k

    def retrieve(self, hyde_queries: List[str]) -> List[Dict[str, Any]]:
        """Run two-stage retrieval on HyDE-expanded queries.

        Parameters
        ----------
        hyde_queries:
            Query list where index 0 is the original cleaned ticket and
            remaining entries are hypothetical expansions.

        Returns
        -------
        list[dict]
            Final top reranked chunks with keys:
            chunk_id, source, text, rerank_score.
        """

        if not hyde_queries:
            return []

        ranked_lists: List[List[Dict[str, Any]]] = []
        for query in hyde_queries:
            ranked_lists.append(self.bm25_retriever.retrieve(query, top_k=self.recall_top_k))
            ranked_lists.append(self.faiss_retriever.retrieve(query, top_k=self.recall_top_k))

        fused_candidates = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k=self.rrf_k,
            top_n=self.rrf_top_n,
        )

        original_query = str(hyde_queries[0])
        return self.reranker.rerank(
            query=original_query,
            candidates=fused_candidates,
            top_k=self.final_top_k,
        )

    def retrieve_for_langgraph(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node adapter that writes retrieval output into graph state.

        Parameters
        ----------
        state:
            Graph state dictionary expected to contain key `hyde_queries`.

        Returns
        -------
        dict
            State update with key `reranked_chunks`.
        """

        hyde_queries = state.get("hyde_queries", [])
        if not isinstance(hyde_queries, list):
            hyde_queries = []
        reranked = self.retrieve([str(q) for q in hyde_queries])
        return {"reranked_chunks": reranked}
