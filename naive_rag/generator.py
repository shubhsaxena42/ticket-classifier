"""
Naive generator: single Groq call, no citation enforcement, no abstention gate.
"""

from __future__ import annotations
from typing import List, Optional
from pathlib import Path
import os

_MODEL = "qwen/qwen3-32b"
_MAX_TOKENS = 512

SYSTEM_PROMPT = """You are a customer support agent.
Answer the user's question using the provided sources.
Be concise and helpful."""

def _load_api_key(env_path: Optional[Path] = None) -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    path = env_path or (Path(__file__).resolve().parents[1] / "backend" / ".env")
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GROQ_API_KEY not set.")


def _build_prompt(query: str, chunks: List[dict]) -> str:
    sources = "\n\n".join(
        f"[{i}] {c['text']}" for i, c in enumerate(chunks)
    )
    return (
        f"Sources:\n{sources}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the sources above."
    )


def _strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    if text.startswith("<think>"):
        text = ""
    return text


def generate(query: str, chunks: List[dict]) -> dict:
    """
    Single Groq call with raw query + retrieved chunks.
    Returns {"answer": str, "model": str, "chunks_used": int}
    No citation check, no retry, no abstention.
    """
    from groq import Groq
    import re

    api_key = _load_api_key()
    client = Groq(api_key=api_key)

    prompt = _build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=_MAX_TOKENS,
    )

    raw = str(response.choices[0].message.content or "").strip()
    raw = _strip_think(raw)

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    return {
        "answer":      raw,
        "model":       _MODEL,
        "chunks_used": len(chunks),
    }
