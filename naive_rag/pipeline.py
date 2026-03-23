"""
NaiveRAGPipeline — end-to-end baseline.

Usage:
    from naive_rag.pipeline import NaiveRAGPipeline

    pipe = NaiveRAGPipeline.build()          # chunks + indexes KB once
    result = pipe.answer("How do I return a product?")
    print(result["answer"])

Run directly for a live comparison vs the advanced pipeline:
    python -m naive_rag.pipeline
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import time
import numpy as np

_DEFAULT_KB = Path(__file__).resolve().parents[1] / "backend" / "Data" / "Raw_Data" / "kb.md"


@dataclass
class NaiveRAGResult:
    query:           str
    retrieved_chunks: List[dict]
    answer:          str
    model:           str
    latency_ms:      float
    num_chunks:      int


@dataclass
class NaiveRAGPipeline:
    chunks:     List[dict]
    embeddings: np.ndarray
    index:      object          # faiss index

    # ── Factory ───────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, kb_path: Optional[Path] = None) -> "NaiveRAGPipeline":
        """Chunk the KB, embed, and build FAISS index. Call once at startup."""
        from naive_rag.chunker import chunk_kb
        from naive_rag.embedder import embed_chunks

        path = kb_path or _DEFAULT_KB
        print(f"[NaiveRAG] Chunking KB: {path}")
        chunks = chunk_kb(path)
        print(f"[NaiveRAG] {len(chunks)} chunks created")

        print("[NaiveRAG] Embedding chunks…")
        embeddings, index = embed_chunks(chunks)
        print(f"[NaiveRAG] Index built ({embeddings.shape[0]} vectors, dim={embeddings.shape[1]})")

        return cls(chunks=chunks, embeddings=embeddings, index=index)

    # ── Core method ───────────────────────────────────────────────────────────
    def answer(self, query: str, top_k: int = 3) -> NaiveRAGResult:
        """
        Naive RAG in three steps:
          1. Embed raw query
          2. FAISS cosine search → top_k chunks
          3. Single Groq call — no HyDE, no reranking, no abstention
        """
        from naive_rag.retriever import retrieve
        from naive_rag.generator import generate

        t0 = time.perf_counter()

        retrieved = retrieve(
            query=query,
            chunks=self.chunks,
            embeddings=self.embeddings,
            faiss_index=self.index,
            top_k=top_k,
        )

        result = generate(query=query, chunks=retrieved)
        latency_ms = (time.perf_counter() - t0) * 1000

        return NaiveRAGResult(
            query=query,
            retrieved_chunks=retrieved,
            answer=result["answer"],
            model=result["model"],
            latency_ms=round(latency_ms, 1),
            num_chunks=len(retrieved),
        )


# ── CLI comparison ────────────────────────────────────────────────────────────
_TEST_QUERIES = [
    "How do I return a defective product?",
    "I was charged twice this month. What should I do?",
    "The app keeps crashing when I try to upload files.",
    "What is your refund policy?",
    "How do I cancel my subscription?",
    "I forgot my password and can't log in.",
    "Does your service support Slack integration?",
    "How long does shipping take?",
]


def _print_result(label: str, result: NaiveRAGResult) -> None:
    print(f"\n{'='*70}")
    print(f"[{label}]  Query: {result.query}")
    print(f"  Latency    : {result.latency_ms:.0f} ms")
    print(f"  Chunks used: {result.num_chunks}")
    print(f"  Top chunk score: {result.retrieved_chunks[0]['retrieval_score']:.3f}"
          if result.retrieved_chunks else "")
    print(f"  Answer:\n    {result.answer[:400]}")


if __name__ == "__main__":
    import sys

    print("\n" + "="*70)
    print(" NAIVE RAG BASELINE")
    print("="*70)
    print("Building index…")
    pipe = NaiveRAGPipeline.build()

    for q in _TEST_QUERIES:
        res = pipe.answer(q)
        _print_result("NAIVE RAG", res)

    print("\n\nDone. Run the advanced pipeline via `python backend/app.py` "
          "and POST to /predict_text for a direct comparison.")
