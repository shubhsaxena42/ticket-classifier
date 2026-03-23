"""
Naive embedder: single sentence-transformer model + FAISS flat index.
No BM25, no hybrid retrieval, no reranking.
"""

from __future__ import annotations
from typing import List
import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


def embed_chunks(chunks: List[dict]) -> tuple[np.ndarray, object]:
    """
    Embed a list of chunk dicts and build a FAISS flat (exact) index.
    Returns (embeddings, faiss_index).
    No quantization, no IVF — just a brute-force cosine search.
    """
    import faiss

    model = _load_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product = cosine on normalized vectors
    index.add(embeddings)

    return embeddings, index


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns a (1, dim) float32 array."""
    model = _load_model()
    vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return vec.astype("float32")
