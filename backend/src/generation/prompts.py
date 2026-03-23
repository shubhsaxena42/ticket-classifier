"""Prompt constants and formatting helpers for response generation.

This module contains generation prompt strings and a single helper function
for formatting retrieved context and ticket text into a user message.
"""

from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT = """
You are a customer support agent. Answer the user's support ticket using
ONLY the information provided in the sources below. Do not use any knowledge
outside of these sources.

Rules:
- Base your response solely on the provided sources
- Be concise and directly address the user's issue
- You MUST cite which source numbers you used
- If the sources do not contain enough information to answer, say so clearly
  and do not invent an answer

You MUST respond with valid JSON matching this exact schema:
{
  "draft_response": "<your response text here>",
  "citations": [<list of integer source indices you used, e.g. 0, 1, 2>]
}

Return ONLY the JSON object. No markdown, no code fences, no extra text.
""".strip()

RETRY_ADDITION = """
IMPORTANT: Your previous response was not valid JSON or did not include citations.
You MUST return valid JSON with this exact format:
{"draft_response": "your answer", "citations": [0, 1]}
The citations array MUST contain at least one source index. This is mandatory.
""".strip()


def build_user_message(ticket: str, chunks: List[Dict[str, Any]]) -> str:
    """Build the generation user message from ticket text and retrieved chunks.

    Args:
        ticket: Cleaned support ticket text.
        chunks: Retrieved chunk dictionaries from reranking.

    Returns:
        A single user message string with numbered source blocks.
    """

    sources_block = ""
    for i, chunk in enumerate(chunks):
        sources_block += f"Source {i}:\n{str(chunk.get('text', ''))}\n\n"

    return (
        f"Sources:\n{sources_block}"
        f"Support ticket:\n{ticket}\n\n"
        "Answer the ticket using only the sources above."
    )
