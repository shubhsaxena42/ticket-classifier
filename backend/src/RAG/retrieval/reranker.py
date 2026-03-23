"""Cross-encoder reranker for final precision ranking in retrieval.

This module defines CrossEncoderReranker, which scores query-chunk pairs with
a cross-encoder and returns the top reranked chunks.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    from sentence_transformers.cross_encoder import CrossEncoder
except ImportError as exc:  # pragma: no cover - exercised in environments without sentence-transformers
    CrossEncoder = None  # type: ignore[assignment]
    _CROSS_ENCODER_IMPORT_ERROR = exc
else:
    _CROSS_ENCODER_IMPORT_ERROR = None


class CrossEncoderReranker:
    """Rerank candidate chunks using a cross-encoder model.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier for the cross-encoder.

    Returns
    -------
    None
        Initializes and stores a loaded cross-encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required to instantiate CrossEncoderReranker."
            ) from _CROSS_ENCODER_IMPORT_ERROR
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Rerank candidate chunks with query-aware cross-encoder logits.

        Parameters
        ----------
        query:
            Original user ticket text used as reranker query.
        candidates:
            Candidate chunks from RRF fusion with chunk metadata.
        top_k:
            Number of top reranked chunks to return.

        Returns
        -------
        list[dict]
            Top reranked chunks with keys: chunk_id, source, text, rerank_score.
        """

        if not candidates:
            return []

        k = max(0, min(int(top_k), len(candidates)))
        if k == 0:
            return []

        pairs = [[query, str(chunk["text"])] for chunk in candidates]
        scores = np.asarray(self.model.predict(pairs), dtype=float)

        scored: List[Dict[str, Any]] = []
        for chunk, score in zip(candidates, scores):
            scored.append(
                {
                    "chunk_id": int(chunk["chunk_id"]),
                    "source": str(chunk["source"]),
                    "text": str(chunk["text"]),
                    "rerank_score": float(score),
                }
            )

        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:k]
