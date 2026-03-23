from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from groq import Groq


DEFAULT_MODEL_ID = "qwen/qwen3-32b"
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7


class TicketState(TypedDict, total=False):
	cleaned_text: str
	queries: List[str]


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


def _build_hyde_prompt(cleaned_text: str) -> str:
	return (
		"Write a formal help desk response for this support ticket. "
		"Use at most 80 tokens.\n"
		f"Ticket: {cleaned_text}"
	)


def _generate_once_sync(
	client: Groq,
	model_id: str,
	prompt: str,
	temperature: float,
	max_output_tokens: int,
) -> str:
	response = client.chat.completions.create(
		model=model_id,
		messages=[{"role": "user", "content": prompt}],
		temperature=temperature,
		max_tokens=max_output_tokens,
	)
	text = str(response.choices[0].message.content or "").strip()
	# Strip <think>...</think> reasoning blocks (qwen3 models).
	if "<think>" in text:
		text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
	# Handle truncated <think> block (no closing tag due to token limit).
	if text.startswith("<think>"):
		text = ""
	return text


async def _generate_two_hypotheticals(
	cleaned_text: str,
	model_id: str,
	temperature: float,
	max_output_tokens: int,
	api_key: str,
) -> List[str]:
	prompt = _build_hyde_prompt(cleaned_text)
	client = Groq(api_key=api_key)

	# Groq SDK call is sync; run both calls in threads for overlap.
	task_1 = asyncio.to_thread(
		_generate_once_sync,
		client,
		model_id,
		prompt,
		temperature,
		max_output_tokens,
	)
	task_2 = asyncio.to_thread(
		_generate_once_sync,
		client,
		model_id,
		prompt,
		temperature,
		max_output_tokens,
	)

	resp_1, resp_2 = await asyncio.gather(task_1, task_2)
	return [resp_1, resp_2]


def _run_async(coro: Any) -> Any:
	try:
		asyncio.get_running_loop()
	except RuntimeError:
		return asyncio.run(coro)

	result: Dict[str, Any] = {}
	error: Dict[str, BaseException] = {}

	def _runner() -> None:
		try:
			result["value"] = asyncio.run(coro)
		except BaseException as exc:
			error["value"] = exc

	import threading

	thread = threading.Thread(target=_runner, daemon=True)
	thread.start()
	thread.join()

	if "value" in error:
		raise error["value"]
	return result.get("value")


def hyde_node(state: TicketState) -> Dict[str, Any]:
	"""Generate HyDE expansions and return state update with `queries`."""
	cleaned_text = str(state.get("cleaned_text", "") or "").strip()
	if not cleaned_text:
		raise ValueError("hyde_node requires non-empty cleaned_text in state.")

	api_key = os.getenv("GROQ_API_KEY", "").strip()
	if not api_key:
		file_key = _read_key_from_env_file(env_path=None, key_name="GROQ_API_KEY")
		api_key = file_key.strip() if file_key else ""
	if not api_key:
		raise RuntimeError(
			"GROQ_API_KEY is not set. HyDE expansion requires a valid Groq API key "
			"in the environment or in backend/.env."
		)

	hypotheticals = _run_async(
		_generate_two_hypotheticals(
			cleaned_text=cleaned_text,
			model_id=DEFAULT_MODEL_ID,
			temperature=DEFAULT_TEMPERATURE,
			max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
			api_key=api_key,
		)
	)

	queries = [cleaned_text]
	queries.extend([text for text in hypotheticals if text])

	# At minimum we need the original query; extra hypotheticals improve recall
	# but are not mandatory.
	return {"queries": queries[:3]}
