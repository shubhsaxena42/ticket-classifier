"""Unit tests for Groq response generator retry and prompt behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generation.generator import ResponseGenerator  # type: ignore[import-not-found]


class _FakeUsage:
    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 5


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]):
        self.usage = _FakeUsage()
        content = json.dumps(payload)

        class _Msg:
            def __init__(self, text: str):
                self.content = text

        class _Choice:
            def __init__(self, text: str):
                self.message = _Msg(text)

        self.choices = [_Choice(content)]


class _FakeClient:
    def __init__(self, payloads: List[Dict[str, Any]]):
        self.payloads = payloads
        self.calls: List[Dict[str, Any]] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self.payloads) - 1)
        return _FakeResponse(self.payloads[idx])


def _make_generator(payloads: List[Dict[str, Any]]) -> ResponseGenerator:
    generator = object.__new__(ResponseGenerator)
    generator.client = _FakeClient(payloads)
    generator.model = "qwen/qwen3-32b"
    generator.max_tokens = 512
    generator.max_retries = 2
    generator.cost_tracker = None
    return generator


def test_generate_with_retry_abstains_after_max_retries() -> None:
    generator = _make_generator(
        [
            {"draft_response": "", "citations": []},
            {"draft_response": "", "citations": []},
            {"draft_response": "", "citations": []},
        ]
    )

    result, did_abstain = generator.generate_with_retry("ticket text", [{"text": "a"}] * 3)

    assert did_abstain is True
    assert result is None
    assert len(generator.client.calls) == 3


def test_generate_returns_on_first_valid_response() -> None:
    generator = _make_generator(
        [
            {"draft_response": "Use source zero", "citations": [0]},
            {"draft_response": "should not be used", "citations": [1]},
        ]
    )

    result, did_abstain = generator.generate_with_retry("ticket text", [{"text": "a"}] * 3)

    assert did_abstain is False
    assert result is not None
    assert result.citations == [0]
    assert len(generator.client.calls) == 1


def test_retry_prompt_includes_retry_addition() -> None:
    generator = _make_generator(
        [
            {"draft_response": "", "citations": []},
            {"draft_response": "Use source one", "citations": [1]},
        ]
    )

    result, did_abstain = generator.generate_with_retry("ticket text", [{"text": "a"}] * 3)

    assert did_abstain is False
    assert result is not None

    first_system = generator.client.calls[0]["messages"][0]["content"]
    second_system = generator.client.calls[1]["messages"][0]["content"]

    assert "IMPORTANT: Your previous response did not include citations." not in first_system
    assert "IMPORTANT: Your previous response did not include citations." in second_system
