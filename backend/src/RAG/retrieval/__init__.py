"""Retrieval package for support ticket RAG.

Exports modular retrieval components and provides a startup factory for loading
artifacts from disk into a ready-to-use TwoStageRetriever.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import faiss  # type: ignore[import-not-found]
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

from .bm25_retriever import BM25Retriever
from .faiss_retriever import FAISSRetriever
from .reranker import CrossEncoderReranker
from .rrf import reciprocal_rank_fusion
from .two_stage_retriever import TwoStageRetriever

PathLike = Union[str, Path]


def _load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load KB chunks from jsonl into a list of metadata dictionaries.

    Parameters
    ----------
    path:
        Path to kb_chunks.jsonl.

    Returns
    -------
    list[dict]
        Chunk metadata list with keys source, chunk_id, and text.
    """

    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def load_retrieval_components(
    models_dir: PathLike,
    data_dir: PathLike,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> TwoStageRetriever:
    """Load retrieval artifacts and return a fully initialized retriever.

    Parameters
    ----------
    models_dir:
        Directory containing bm25_index.pkl and faiss_index.bin.
    data_dir:
        Directory containing kb_chunks.jsonl.
    embed_model_name:
        SentenceTransformer model name for dense query embeddings.

    Returns
    -------
    TwoStageRetriever
        Initialized two-stage retriever ready for inference.
    """

    if faiss is None:
        raise ImportError(
            "faiss is required to load retrieval components. Install dependency: faiss-cpu"
        ) from _FAISS_IMPORT_ERROR
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required to load retrieval components."
        ) from _ST_IMPORT_ERROR

    models_dir_path = Path(models_dir)
    data_dir_path = Path(data_dir)

    chunks_path = data_dir_path / "kb_chunks.jsonl"
    bm25_path = models_dir_path / "bm25_index.pkl"
    faiss_path = models_dir_path / "faiss_index.bin"

    chunks = _load_chunks_jsonl(chunks_path)

    with bm25_path.open("rb") as handle:
        bm25_bundle = pickle.load(handle)
    bm25_index = bm25_bundle["index"]

    faiss_index = faiss.read_index(str(faiss_path))
    embed_model = SentenceTransformer(embed_model_name)

    bm25_retriever = BM25Retriever(index=bm25_index, chunks=chunks)
    faiss_retriever = FAISSRetriever(index=faiss_index, chunks=chunks, embed_model=embed_model)
    reranker = CrossEncoderReranker()

    return TwoStageRetriever(
        bm25_retriever=bm25_retriever,
        faiss_retriever=faiss_retriever,
        reranker=reranker,
    )


__all__ = [
    "BM25Retriever",
    "FAISSRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "TwoStageRetriever",
    "load_retrieval_components",
]
