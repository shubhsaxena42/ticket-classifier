"""Reciprocal Rank Fusion for combining multiple retrieval result lists.

This module provides reciprocal_rank_fusion, a model-agnostic rank aggregation
function that merges and deduplicates chunks across ranked lists.
"""

from __future__ import annotations

from typing import Any, Dict, List


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_n: int = 30,
) -> List[Dict[str, Any]]:
    """Fuse ranked lists with Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    ranked_lists:
        A list of ranked retrieval lists; each inner list contains chunk dicts.
    k:
        RRF smoothing constant. Standard default is 60.
    top_n:
        Maximum number of fused chunks to return.

    Returns
    -------
    list[dict]
        Deduplicated chunks sorted by descending RRF score with keys:
        chunk_id, source, text, rrf_score.
    """

    if top_n <= 0:
        return []

    score_by_chunk_id: Dict[int, float] = {}
    metadata_by_chunk_id: Dict[int, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            chunk_id = int(chunk["chunk_id"])
            score_by_chunk_id[chunk_id] = score_by_chunk_id.get(chunk_id, 0.0) + (
                1.0 / (k + rank + 1)
            )
            if chunk_id not in metadata_by_chunk_id:
                metadata_by_chunk_id[chunk_id] = {
                    "chunk_id": chunk_id,
                    "source": str(chunk["source"]),
                    "text": str(chunk["text"]),
                }

    ranked_chunk_ids = sorted(
        score_by_chunk_id.keys(),
        key=lambda cid: score_by_chunk_id[cid],
        reverse=True,
    )

    output: List[Dict[str, Any]] = []
    for chunk_id in ranked_chunk_ids[:top_n]:
        chunk_meta = metadata_by_chunk_id[chunk_id]
        output.append(
            {
                "chunk_id": int(chunk_meta["chunk_id"]),
                "source": str(chunk_meta["source"]),
                "text": str(chunk_meta["text"]),
                "rrf_score": float(score_by_chunk_id[chunk_id]),
            }
        )
    return output
