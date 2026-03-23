"""BM25 sparse retrieval component for support ticket RAG.

This module defines BM25Retriever, a lightweight wrapper around BM25Okapi
that tokenizes queries consistently and returns scored chunk metadata.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover - exercised in environments without rank_bm25
    BM25Okapi = None  # type: ignore[assignment]
    _BM25_IMPORT_ERROR = exc
else:
    _BM25_IMPORT_ERROR = None


class BM25Retriever:
    """Retrieve chunks using BM25 lexical similarity.

    Parameters
    ----------
    index:
        Preloaded BM25Okapi index created during offline indexing.
    chunks:
        Chunk metadata list. Each item should include chunk_id, source, and text.

    Returns
    -------
    None
        Initializes a retriever instance for repeated query-time use.
    """

    def __init__(self, index: BM25Okapi, chunks: List[Dict[str, Any]]):
        if BM25Okapi is None:
            raise ImportError(
                "rank_bm25 is required to instantiate BM25Retriever. "
                "Install dependency: rank-bm25"
            ) from _BM25_IMPORT_ERROR
        self.index = index
        self.chunks = chunks

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using the same lowercase-regex rule as indexing.

        Parameters
        ----------
        text:
            Input text to tokenize.

        Returns
        -------
        list[str]
            Lowercased word tokens extracted via regex.
        """

        return re.findall(r"\b\w+\b", str(text).lower())

    def retrieve(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Return top-k BM25-scored chunks for a query.

        Parameters
        ----------
        query:
            Query string to score against all chunks.
        top_k:
            Number of ranked chunks to return.

        Returns
        -------
        list[dict]
            Ranked chunk dictionaries with keys: chunk_id, source, text, score.
        """

        if not self.chunks:
            return []

        tokens = self.tokenize(query)
        scores = np.asarray(self.index.get_scores(tokens), dtype=float)

        k = max(0, min(int(top_k), len(self.chunks)))
        if k == 0:
            return []

        top_indices = np.argsort(scores)[-k:][::-1]
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": int(chunk["chunk_id"]),
                    "source": str(chunk["source"]),
                    "text": str(chunk["text"]),
                    "score": float(scores[int(idx)]),
                }
            )
        return results
