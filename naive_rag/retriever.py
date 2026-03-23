"""
Naive retriever: single-stage FAISS cosine search.
No BM25, no RRF fusion, no cross-encoder reranking.
"""

from __future__ import annotations
from typing import List
import numpy as np

TOP_K = 3   # how many chunks to return to the generator


def retrieve(
    query: str,
    chunks: List[dict],
    embeddings: np.ndarray,
    faiss_index,
    top_k: int = TOP_K,
) -> List[dict]:
    """
    Embed the raw query and return the top_k most similar chunks.
    No query expansion, no reranking — pure cosine similarity.
    """
    from naive_rag.embedder import embed_query

    query_vec = embed_query(query)                          # (1, dim)
    scores, indices = faiss_index.search(query_vec, top_k) # brute-force search

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["retrieval_score"] = float(score)
        results.append(chunk)

    return results
