"""FAISS dense retrieval component for support ticket RAG.

This module defines FAISSRetriever, which embeds queries with a preloaded
SentenceTransformer and searches a preloaded FAISS index.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - exercised in environments without faiss
    faiss = None  # type: ignore[assignment]
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - exercised in environments without sentence-transformers
    SentenceTransformer = None  # type: ignore[assignment]
    _ST_IMPORT_ERROR = exc
else:
    _ST_IMPORT_ERROR = None


class FAISSRetriever:
    """Retrieve chunks using dense semantic similarity over a FAISS index.

    Parameters
    ----------
    index:
        Preloaded FAISS IndexFlatIP built on normalized embeddings.
    chunks:
        Chunk metadata list where list position matches FAISS row id.
    embed_model:
        Preloaded sentence-transformers model used for query embeddings.

    Returns
    -------
    None
        Initializes a retriever instance for repeated query-time use.
    """

    def __init__(self, index: Any, chunks: List[Dict[str, Any]], embed_model: Any):
        if faiss is None:
            raise ImportError(
                "faiss is required to instantiate FAISSRetriever. "
                "Install dependency: faiss-cpu"
            ) from _FAISS_IMPORT_ERROR
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required to instantiate FAISSRetriever."
            ) from _ST_IMPORT_ERROR
        self.index = index
        self.chunks = chunks
        self.embed_model = embed_model

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed text inputs into normalized float32 vectors.

        Parameters
        ----------
        texts:
            List of text strings to embed.

        Returns
        -------
        np.ndarray
            Embedding matrix with shape (len(texts), embedding_dim), float32.
        """

        embeddings = self.embed_model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Return top-k FAISS nearest chunks for a query.

        Parameters
        ----------
        query:
            Query string for dense retrieval.
        top_k:
            Number of neighbors to retrieve.

        Returns
        -------
        list[dict]
            Ranked chunk dictionaries with keys: chunk_id, source, text, score.
        """

        if not self.chunks:
            return []

        k = max(0, min(int(top_k), len(self.chunks)))
        if k == 0:
            return []

        query_embedding = self.embed([query])
        scores, indices = self.index.search(query_embedding, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) == -1:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": int(chunk["chunk_id"]),
                    "source": str(chunk["source"]),
                    "text": str(chunk["text"]),
                    "score": float(score),
                }
            )
        return results
