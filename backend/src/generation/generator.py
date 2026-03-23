"""Groq-based structured response generation with citation-aware retry logic."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq
from pydantic import ValidationError

from evaluation.cost_tracker import CostTracker
from generation.abstention_gate import ABSTAIN_RESPONSE_TEXT
from generation.prompts import RETRY_ADDITION, SYSTEM_PROMPT, build_user_message
from generation.schemas import GenerationOutput

GenerationResult = Tuple[Optional[GenerationOutput], bool]

DEFAULT_GENERATION_MODEL = "qwen/qwen3-32b"
FALLBACK_GENERATION_MODELS = (
    "qwen/qwen3-32b",
)


def _default_env_path() -> Path:
    """Return backend .env path used for local API-key fallback loading."""

    return Path(__file__).resolve().parents[2] / ".env"


def _read_key_from_env_file(env_path: Optional[Path], key_name: str) -> Optional[str]:
    """Read one key from a simple KEY=VALUE env file.

    Args:
        env_path: Optional custom env file path.
        key_name: Variable name to resolve.

    Returns:
        The resolved string value or None when not found.
    """

    path = env_path if env_path is not None else _default_env_path()
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key_name:
            return v.strip().strip('"').strip("'")
    return None


class ResponseGenerator:
    """Generate grounded ticket responses from retrieved chunks using Groq.

    Args:
        groq_api_key: Optional explicit API key. Falls back to GROQ_API_KEY.
        model: Groq model id for generation.
        max_tokens: Maximum output tokens for generation responses.
        max_retries: Number of retries after the first attempt for citation failures.
        cost_tracker: Optional usage and cost tracker.

    Returns:
        None. Use generate_with_retry() or generate_for_langgraph().
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        model: str = DEFAULT_GENERATION_MODEL,
        max_tokens: int = 1024,
        max_retries: int = 2,
        cost_tracker: Optional[CostTracker] = None,
    ):
        key = (groq_api_key or gemini_api_key or os.environ.get("GROQ_API_KEY", "") or "").strip()
        if not key:
            file_key = _read_key_from_env_file(env_path=None, key_name="GROQ_API_KEY")
            key = file_key.strip() if file_key else ""
        if not key:
            raise RuntimeError("GROQ_API_KEY is required for ResponseGenerator.")

        self.client = Groq(api_key=key)
        self.model = model
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self.cost_tracker = cost_tracker

    @staticmethod
    def _build_model_candidates(primary_model: str) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for model_name in (primary_model, *FALLBACK_GENERATION_MODELS):
            key = str(model_name).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    def generate(
        self,
        ticket: str,
        chunks: List[Dict[str, Any]],
        is_retry: bool = False,
        cost_tracker: Optional[CostTracker] = None,
    ) -> GenerationOutput:
        """Generate one structured response attempt from ticket and chunks.

        Args:
            ticket: Cleaned ticket text.
            chunks: Retrieved chunk dictionaries used as sources.
            is_retry: Whether this call is a retry attempt.
            cost_tracker: Optional per-call tracker override.

        Returns:
            Validated GenerationOutput.

        Raises:
            ValidationError: When model output violates GenerationOutput schema.
            json.JSONDecodeError: When response text is not valid JSON.
        """

        system = SYSTEM_PROMPT
        if is_retry:
            system = f"{SYSTEM_PROMPT}\n\n{RETRY_ADDITION}"

        user_message = build_user_message(ticket=ticket, chunks=chunks)

        response: Any = None
        last_error: Optional[Exception] = None
        for model_name in self._build_model_candidates(self.model):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                )
                break
            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()
                if "not found" in error_text or "model" in error_text:
                    continue
                if "rate" in error_text or "quota" in error_text:
                    break
                break

        if response is None:
            if last_error is not None:
                raise RuntimeError(f"Generation failed: {last_error}")
            raise RuntimeError("Generation failed without a response.")

        tracker = cost_tracker or self.cost_tracker
        usage_meta = getattr(response, "usage", None)
        if tracker is not None and usage_meta is not None:
            input_tokens = int(getattr(usage_meta, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage_meta, "completion_tokens", 0) or 0)
            tracker.log(
                component="generation",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        response_text = str(response.choices[0].message.content or "").strip()
        # Strip <think>...</think> reasoning blocks (qwen3 models).
        if "<think>" in response_text:
            response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()
        # Handle truncated <think> block (no closing tag due to token limit).
        if response_text.startswith("<think>"):
            raise ValueError("Groq response was truncated inside <think> block.")
        if response_text.startswith("```"):
            response_text = response_text.strip("`")
            if response_text.lower().startswith("json"):
                response_text = response_text[4:].strip()
        payload = json.loads(response_text)
        return GenerationOutput.model_validate(payload)

    def generate_with_retry(
        self,
        ticket: str,
        chunks: List[Dict[str, Any]],
        cost_tracker: Optional[CostTracker] = None,
    ) -> GenerationResult:
        """Generate with retries until a citation-bearing response is obtained.

        Args:
            ticket: Cleaned ticket text.
            chunks: Retrieved chunks used as context.
            cost_tracker: Optional per-call tracker override.

        Returns:
            Tuple of (GenerationOutput or None, did_abstain boolean).
        """

        for attempt in range(self.max_retries + 1):
            is_retry = attempt > 0
            try:
                result = self.generate(
                    ticket=ticket,
                    chunks=chunks,
                    is_retry=is_retry,
                    cost_tracker=cost_tracker,
                )
            except (
                ValidationError,
                json.JSONDecodeError,
                ValueError,
                RuntimeError,
            ):
                continue

            if len(result.citations) > 0:
                return result, False

        return None, True

    def generate_for_langgraph(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node adapter for response generation state updates.

        Args:
            state: Pipeline state with cleaned_message and reranked_chunks.

        Returns:
            Partial state update containing abstention/generation fields.
        """

        cleaned_message = str(state.get("cleaned_message", "") or "")
        reranked_chunks = state.get("reranked_chunks", [])
        if not isinstance(reranked_chunks, list):
            reranked_chunks = []

        result, did_abstain = self.generate_with_retry(
            ticket=cleaned_message,
            chunks=reranked_chunks,
        )

        if did_abstain or result is None:
            return {
                "abstain_flag": True,
                "draft_response": ABSTAIN_RESPONSE_TEXT,
                "citations": [],
                "top_3_sources": [str(chunk.get("source", "")) for chunk in reranked_chunks],
            }

        return {
            "abstain_flag": False,
            "draft_response": result.draft_response,
            "citations": list(result.citations),
            "top_3_sources": [str(chunk.get("source", "")) for chunk in reranked_chunks],
        }
