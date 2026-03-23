"""Unit tests for the two-stage retrieval module.

These tests validate data shapes, ordering, deduplication, and end-to-end
behavior for BM25, FAISS, RRF, reranking, and LangGraph adapter output.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from RAG.retrieval.bm25_retriever import BM25Retriever
from RAG.retrieval.faiss_retriever import FAISSRetriever
from RAG.retrieval.reranker import CrossEncoderReranker
from RAG.retrieval.rrf import reciprocal_rank_fusion
from RAG.retrieval.two_stage_retriever import TwoStageRetriever


class _DummyEmbedModel:
    """Deterministic embedding model for tests.

    Parameters
    ----------
    mapping:
        Dictionary mapping text to embedding vectors.

    Returns
    -------
    None
        Stores deterministic vectors for encode() calls.
    """

    def __init__(self, mapping: Dict[str, np.ndarray]):
        self.mapping = mapping

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for text in texts:
            vec = np.asarray(self.mapping.get(text, np.zeros(4, dtype=np.float32)), dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors)


class _DummyCrossEncoder:
    """Minimal CrossEncoder substitute returning deterministic scores."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict(self, pairs: List[List[str]]) -> np.ndarray:
        # Score by chunk text length to produce stable ranking in tests.
        return np.asarray([len(pair[1]) / 10.0 for pair in pairs], dtype=np.float32)


def _build_mock_chunks(n: int = 5) -> List[Dict[str, Any]]:
    return [
        {"source": "mock_kb.json", "chunk_id": i, "text": text}
        for i, text in enumerate(
            [
                "password reset steps and troubleshooting",
                "billing refund policy and invoice checks",
                "account login and authentication issue guide",
                "shipping delay and tracking support",
                "subscription cancellation process",
            ][:n]
        )
    ]


def test_bm25_returns_correct_format() -> None:
    rank_bm25 = pytest.importorskip("rank_bm25")

    chunks = _build_mock_chunks(5)
    tokenized_corpus = [
        BM25Retriever.__dict__["tokenize"](BM25Retriever.__new__(BM25Retriever), chunk["text"])
        for chunk in chunks
    ]
    index = rank_bm25.BM25Okapi(tokenized_corpus)
    retriever = BM25Retriever(index=index, chunks=chunks)

    results = retriever.retrieve("password reset", top_k=3)

    assert len(results) == 3
    for item in results:
        assert set(item.keys()) == {"chunk_id", "source", "text", "score"}
    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True)


def test_faiss_returns_correct_format() -> None:
    faiss = pytest.importorskip("faiss")

    chunks = _build_mock_chunks(5)
    base_vectors = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    base_vectors /= np.linalg.norm(base_vectors, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(base_vectors.shape[1])
    index.add(base_vectors)

    mapping = {"cannot log in": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    embed_model = _DummyEmbedModel(mapping=mapping)

    retriever = FAISSRetriever(index=index, chunks=chunks, embed_model=embed_model)
    results = retriever.retrieve("cannot log in", top_k=3)

    assert len(results) == 3
    for item in results:
        assert set(item.keys()) == {"chunk_id", "source", "text", "score"}
        assert -1.0 <= item["score"] <= 1.0


def test_rrf_deduplicates() -> None:
    list1 = [
        {"chunk_id": 1, "source": "a", "text": "x", "score": 1.0},
        {"chunk_id": 2, "source": "a", "text": "y", "score": 0.8},
    ]
    list2 = [
        {"chunk_id": 1, "source": "a", "text": "x", "score": 0.9},
        {"chunk_id": 3, "source": "b", "text": "z", "score": 0.7},
    ]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=10)

    ids = [item["chunk_id"] for item in fused]
    assert len(ids) == len(set(ids))

    scores_by_id = {item["chunk_id"]: item["rrf_score"] for item in fused}
    assert scores_by_id[1] > scores_by_id[2]


def test_rrf_ordering() -> None:
    list1 = [
        {"chunk_id": 5, "source": "a", "text": "best", "score": 1.0},
        {"chunk_id": 1, "source": "a", "text": "x", "score": 0.8},
    ]
    list2 = [
        {"chunk_id": 5, "source": "a", "text": "best", "score": 0.95},
        {"chunk_id": 2, "source": "a", "text": "y", "score": 0.7},
    ]
    list3 = [
        {"chunk_id": 5, "source": "a", "text": "best", "score": 0.91},
        {"chunk_id": 3, "source": "a", "text": "z", "score": 0.6},
    ]

    fused = reciprocal_rank_fusion([list1, list2, list3], k=60, top_n=10)
    assert fused[0]["chunk_id"] == 5


def test_reranker_uses_original_query(monkeypatch: pytest.MonkeyPatch) -> None:
    import RAG.retrieval.reranker as reranker_module

    monkeypatch.setattr(reranker_module, "CrossEncoder", _DummyCrossEncoder)
    reranker = CrossEncoderReranker()

    candidates = [
        {"chunk_id": 1, "source": "a", "text": "short"},
        {"chunk_id": 2, "source": "a", "text": "a much longer chunk text"},
        {"chunk_id": 3, "source": "a", "text": "medium length"},
        {"chunk_id": 4, "source": "a", "text": "tiny"},
        {"chunk_id": 5, "source": "a", "text": "quite descriptive answer chunk"},
    ]

    results = reranker.rerank(query="original ticket", candidates=candidates, top_k=3)

    assert len(results) == 3
    for item in results:
        assert "rerank_score" in item

    scores = [item["rerank_score"] for item in results]
    assert scores == sorted(scores, reverse=True)


def test_two_stage_retriever_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rank_bm25 = pytest.importorskip("rank_bm25")
    faiss = pytest.importorskip("faiss")

    import RAG.retrieval as retrieval_pkg
    import RAG.retrieval.reranker as reranker_module

    monkeypatch.setattr(reranker_module, "CrossEncoder", _DummyCrossEncoder)

    chunks = [
        {"source": "fixture.json", "chunk_id": i, "text": f"chunk text {i} about login auth reset"}
        for i in range(10)
    ]

    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    kb_chunks_path = data_dir / "kb_chunks.jsonl"
    with kb_chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")

    tokenized_corpus = [
        BM25Retriever.__dict__["tokenize"](BM25Retriever.__new__(BM25Retriever), chunk["text"])
        for chunk in chunks
    ]
    bm25_index = rank_bm25.BM25Okapi(tokenized_corpus)
    with (models_dir / "bm25_index.pkl").open("wb") as handle:
        pickle.dump({"index": bm25_index, "chunks": chunks}, handle)

    vecs = np.asarray([[1.0, 0.0, 0.0, float(i) / 20.0] for i in range(10)], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(models_dir / "faiss_index.bin"))

    class _FactoryEmbedModel(_DummyEmbedModel):
        pass

    monkeypatch.setattr(retrieval_pkg, "SentenceTransformer", lambda _: _FactoryEmbedModel({
        "login broken": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "cannot authenticate": np.asarray([0.95, 0.05, 0.0, 0.0], dtype=np.float32),
        "password reset": np.asarray([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
    }))

    retriever = retrieval_pkg.load_retrieval_components(models_dir=models_dir, data_dir=data_dir)
    result = retriever.retrieve(
        hyde_queries=["login broken", "cannot authenticate", "password reset"]
    )

    assert len(result) == 3
    for item in result:
        assert set(item.keys()) == {"chunk_id", "source", "text", "rerank_score"}
    scores = [item["rerank_score"] for item in result]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_for_langgraph_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    import RAG.retrieval.reranker as reranker_module

    monkeypatch.setattr(reranker_module, "CrossEncoder", _DummyCrossEncoder)

    class _LocalBM25:
        def retrieve(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
            return [
                {"chunk_id": 1, "source": "x", "text": f"bm25-1 {query}", "score": 1.0},
                {"chunk_id": 2, "source": "x", "text": f"bm25-2 {query}", "score": 0.95},
            ]

    class _LocalFAISS:
        def retrieve(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
            return [
                {"chunk_id": 3, "source": "x", "text": f"faiss-1 {query}", "score": 0.9},
                {"chunk_id": 4, "source": "x", "text": f"faiss-2 {query}", "score": 0.85},
            ]

    retriever = TwoStageRetriever(
        bm25_retriever=_LocalBM25(),  # type: ignore[arg-type]
        faiss_retriever=_LocalFAISS(),  # type: ignore[arg-type]
        reranker=CrossEncoderReranker(),
        final_top_k=3,
    )

    output = retriever.retrieve_for_langgraph(
        {"hyde_queries": ["test query", "hyp1", "hyp2"]}
    )

    assert isinstance(output, dict)
    assert "reranked_chunks" in output
    assert isinstance(output["reranked_chunks"], list)
    assert len(output["reranked_chunks"]) == 3
