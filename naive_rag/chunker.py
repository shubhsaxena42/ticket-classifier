"""
Naive chunker: fixed-size, no overlap, paragraph-aware splitting.
No semantic boundary detection, no sliding window.
"""

from __future__ import annotations
from pathlib import Path
from typing import List


CHUNK_WORDS = 120       # target words per chunk (hard ceiling)
MIN_CHUNK_WORDS = 20    # discard chunks shorter than this


def _split_paragraphs(text: str) -> List[str]:
    """Split on blank lines; preserve non-empty paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _chunk_paragraph(para: str, max_words: int) -> List[str]:
    """If a paragraph exceeds max_words, split at sentence boundaries."""
    words = para.split()
    if len(words) <= max_words:
        return [para]

    chunks, current = [], []
    for sentence in para.replace("?", "?.").replace("!", "!.").split(". "):
        sentence_words = sentence.split()
        if len(current) + len(sentence_words) > max_words and current:
            chunks.append(" ".join(current))
            current = sentence_words
        else:
            current.extend(sentence_words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_kb(kb_path: Path | str) -> List[dict]:
    """
    Read a markdown KB file and return a flat list of chunk dicts.
    Each chunk: {"chunk_id": int, "text": str, "source": str}
    """
    text = Path(kb_path).read_text(encoding="utf-8")
    paragraphs = _split_paragraphs(text)

    chunks: List[dict] = []
    for para in paragraphs:
        for raw_chunk in _chunk_paragraph(para, CHUNK_WORDS):
            words = raw_chunk.split()
            if len(words) < MIN_CHUNK_WORDS:
                continue
            chunks.append({
                "chunk_id": len(chunks),
                "text": raw_chunk,
                "source": str(kb_path),
            })

    return chunks
