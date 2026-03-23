from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from groq import Groq
from pydantic import ValidationError

from Scrubber.Rules import TicketPrediction


DEFAULT_MODEL_ID = "qwen/qwen3-32b"
DEFAULT_MAX_OUTPUT_TOKENS = 1024
FALLBACK_MODEL_IDS = (
	"qwen/qwen3-32b",
)


SYSTEM_PROMPT = """
You are Tier 3, the final arbitration classifier for support tickets.

Strict Grounding Rules:
1. Use ONLY the provided Knowledge Base (KB) and ticket text.
2. Do not invent policies, products, classes, or priorities not present in the KB.
3. If evidence is weak, choose conservative routing by setting abstain_flag=true.
4. Return only schema-conformant JSON.
""".strip()


def _default_kb_path() -> Path:
	return Path(__file__).resolve().parents[2] / "Data" / "Raw_Data" / "kb.md"


def _default_env_path() -> Path:
	return Path(__file__).resolve().parents[2] / ".env"


def _read_key_from_env_file(env_path: Optional[Path], key_name: str) -> Optional[str]:
	path = env_path if env_path is not None else _default_env_path()
	if not path.exists():
		return None

	for raw_line in path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		k, v = line.split("=", 1)
		if k.strip() != key_name:
			continue
		return v.strip().strip('"').strip("'")
	return None


def _load_kb_text(kb_text: Optional[str] = None, kb_path: Optional[Path] = None) -> str:
	if kb_text is not None and kb_text.strip():
		return kb_text.strip()

	path = kb_path if kb_path is not None else _default_kb_path()
	if not path.exists():
		raise FileNotFoundError(f"Knowledge base file not found: {path}")
	return path.read_text(encoding="utf-8")


_VALID_CATEGORIES = [
	"Billing inquiry",
	"Cancellation request",
	"Product inquiry",
	"Refund request",
	"Technical issue",
]

_VALID_PRIORITIES = ["Critical", "High", "Medium", "Low"]


_MAX_KB_CHARS = 3000


def _build_user_prompt(cleaned_text: str, kb_content: str) -> str:
	categories_str = ", ".join(f'"{c}"' for c in _VALID_CATEGORIES)
	priorities_str = ", ".join(f'"{p}"' for p in _VALID_PRIORITIES)
	# Truncate KB to stay within token limits — classification only needs
	# category/priority signals, not full retrieval context.
	kb_snippet = kb_content[:_MAX_KB_CHARS] if len(kb_content) > _MAX_KB_CHARS else kb_content
	return (
		"Classify this support ticket into one of the given categories and priorities.\n\n"
		f"Ticket:\n{cleaned_text}\n\n"
		f"Reference knowledge base (excerpt):\n{kb_snippet}\n\n"
		f"Valid categories (pick exactly one): [{categories_str}]\n"
		f"Valid priorities (pick exactly one): [{priorities_str}]\n\n"
		"Return JSON with fields: "
		"predicted_category, predicted_priority, confidence_score, tier_used, abstain_flag, routing_action.\n"
		"predicted_category MUST be one of the valid categories listed above.\n"
		"predicted_priority MUST be one of the valid priorities listed above.\n"
		"For this node set tier_used to 'groq'.\n"
		"If confidence is low, set abstain_flag=true and routing_action='manual_review'."
	)


_VALID_ROUTING_ACTIONS = {"auto_route", "suggest", "manual_review", "escalate_to_tier2", "escalate_to_tier3"}


def _parse_prediction_from_response_text(response_text: str) -> TicketPrediction:
	obj = json.loads(response_text)
	# Normalize routing_action — the LLM may invent values like "respond_with_kb".
	raw_action = str(obj.get("routing_action", "")).strip().lower()
	if raw_action not in _VALID_ROUTING_ACTIONS:
		# Map based on abstain_flag and confidence.
		abstain = obj.get("abstain_flag", False)
		conf = float(obj.get("confidence_score", 0) or 0)
		if abstain:
			obj["routing_action"] = "manual_review"
		elif conf >= 0.90:
			obj["routing_action"] = "auto_route"
		elif conf >= 0.65:
			obj["routing_action"] = "suggest"
		else:
			obj["routing_action"] = "manual_review"
	return TicketPrediction.model_validate(obj)


def _extract_response_text(response: object) -> str:
	text = str(getattr(response, "text", "") or "").strip()
	if text:
		return text

	candidates = getattr(response, "candidates", None)
	if not candidates:
		return ""

	try:
		parts = candidates[0].content.parts
		if not parts:
			return ""
		return str(getattr(parts[0], "text", "") or "").strip()
	except (IndexError, AttributeError, TypeError):
		return ""


class Tier3Error(RuntimeError):
	"""Raised when Tier 3 Groq arbitration cannot produce a real prediction."""


def _build_model_candidates(primary_model_id: str) -> list[str]:
	seen = set()
	ordered: list[str] = []
	for model_id in (primary_model_id, *FALLBACK_MODEL_IDS):
		model_key = str(model_id).strip()
		if not model_key or model_key in seen:
			continue
		seen.add(model_key)
		ordered.append(model_key)
	return ordered

def groq_tier3_node(
	cleaned_text: str,
	kb_text: Optional[str] = None,
	kb_path: Optional[Path] = None,
	env_path: Optional[Path] = None,
	model_id: str = DEFAULT_MODEL_ID,
	max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
	max_retries: int = 2,
) -> TicketPrediction:
	"""Run Tier 3 Groq arbitration and return a validated TicketPrediction."""
	api_key = os.getenv("GROQ_API_KEY", "").strip()
	if not api_key:
		file_key = _read_key_from_env_file(env_path=env_path, key_name="GROQ_API_KEY")
		api_key = file_key.strip() if file_key else ""
	if not api_key:
		raise Tier3Error(
			"GROQ_API_KEY is not set. Tier 3 requires a valid Groq API key "
			"in the environment or in backend/.env."
		)

	client = Groq(api_key=api_key)
	kb_content = _load_kb_text(kb_text=kb_text, kb_path=kb_path)
	model_candidates = _build_model_candidates(model_id)

	prompt = _build_user_prompt(cleaned_text=str(cleaned_text).strip(), kb_content=kb_content)
	last_error: Optional[Exception] = None

	for candidate_model in model_candidates:
		for _ in range(max_retries + 1):
			try:
				response = client.chat.completions.create(
					model=candidate_model,
					messages=[
						{"role": "system", "content": SYSTEM_PROMPT},
						{"role": "user", "content": prompt},
					],
					temperature=0.1,
					max_tokens=max_output_tokens,
				)
				response_text = str(response.choices[0].message.content or "").strip()
				# Strip <think>...</think> reasoning blocks (qwen3 models).
				if "<think>" in response_text:
					response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()
				# Handle truncated <think> block (no closing tag due to token limit).
				if response_text.startswith("<think>"):
					response_text = ""
				if response_text.startswith("```"):
					response_text = response_text.strip("`")
					if response_text.lower().startswith("json"):
						response_text = response_text[4:].strip()
				if not response_text:
					raise ValueError("Groq returned an empty response.")

				prediction = _parse_prediction_from_response_text(response_text)

				prediction.tier_used = f"groq:{candidate_model}"

				return prediction
			except (json.JSONDecodeError, ValidationError, ValueError, Exception) as exc:
				last_error = exc
				error_text = str(exc)
				if "rate" in error_text.lower() or "quota" in error_text.lower():
					break
				if "NOT_FOUND" in error_text or "not found" in error_text.lower():
					break
				time.sleep(0.3)

	raise Tier3Error(
		f"Tier 3 Groq arbitration failed after exhausting all model candidates "
		f"({', '.join(model_candidates)}) with {max_retries} retries each. "
		f"Last error: {last_error}"
	)


def gemini_tier3_node(
	cleaned_text: str,
	kb_text: Optional[str] = None,
	kb_path: Optional[Path] = None,
	env_path: Optional[Path] = None,
	model_id: str = DEFAULT_MODEL_ID,
	max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
	max_retries: int = 2,
) -> TicketPrediction:
	"""Backward-compatible alias for Groq-based Tier 3 arbitration."""
	return groq_tier3_node(
		cleaned_text=cleaned_text,
		kb_text=kb_text,
		kb_path=kb_path,
		env_path=env_path,
		model_id=model_id,
		max_output_tokens=max_output_tokens,
		max_retries=max_retries,
	)
