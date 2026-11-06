"""Pluggable LLM clients used as the *teacher* (writes data) and *judge* (scores data).

Backends:
  anthropic  - Claude via the official SDK (default)
  openai     - any OpenAI-compatible server, e.g. a local vLLM/Ollama running an open model.
               Useful if you prefer an open-weights teacher for licensing reasons.
  mock       - deterministic fake for tests and dry runs; no network.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, str]  # {"role": "user"|"assistant", "content": str}


class TeacherError(RuntimeError):
    pass


class Teacher(Protocol):
    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 2048,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Return the assistant text. With `schema`, the text is JSON matching it."""
        ...


@dataclass
class TeacherConfig:
    backend: str = "anthropic"
    model: str = "claude-opus-5-5"
    effort: str | None = "medium"
    base_url: str | None = None
    api_key_env: str | None = None
    max_retries: int = 4
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- Anthropic
class AnthropicTeacher:
    def __init__(self, cfg: TeacherConfig):
        import anthropic

        self._anthropic = anthropic
        kwargs: dict[str, Any] = {"max_retries": cfg.max_retries}
        if cfg.api_key_env:
            kwargs["api_key"] = os.environ[cfg.api_key_env]
        self.client = anthropic.Anthropic(**kwargs)
        self.cfg = cfg

    def complete(self, system, messages, *, max_tokens=2048, schema=None):
        output_config: dict[str, Any] = {}
        if self.cfg.effort:
            output_config["effort"] = self.cfg.effort
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        # `fallbacks: "default"` re-runs a request a safety classifier declined on
        # Anthropic's recommended fallback model, instead of returning a refusal.
        resp = self.client.beta.messages.create(
            model=self.cfg.model,
            max_tokens=max_tokens,
            # The persona/judge system prompt is identical across thousands of calls: cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config=output_config or self._anthropic.omit,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            raise TeacherError(f"refused: {getattr(resp.stop_details, 'category', None)}")
        if resp.stop_reason == "max_tokens":
            raise TeacherError("hit max_tokens; increase it")
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not text:
            raise TeacherError("empty response")
        return text


# --------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatTeacher:
    def __init__(self, cfg: TeacherConfig):
        from openai import OpenAI

        key = os.environ.get(cfg.api_key_env or "OPENAI_API_KEY", "not-needed")
        self.client = OpenAI(base_url=cfg.base_url, api_key=key, max_retries=cfg.max_retries)
        self.cfg = cfg

    def complete(self, system, messages, *, max_tokens=2048, schema=None):
        kwargs: dict[str, Any] = {}
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema},
            }
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
            **kwargs,
            **self.cfg.extra,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise TeacherError("empty response")
        return text


# --------------------------------------------------------------------------- Mock
class MockTeacher:
    """Produces plausible-shaped output so the whole pipeline runs offline."""

    def __init__(self, cfg: TeacherConfig | None = None, seed: int = 0):
        self.rng = random.Random(seed)
        self.calls = 0

    def complete(self, system, messages, *, max_tokens=2048, schema=None):
        self.calls += 1
        last = messages[-1]["content"] if messages else ""
        if schema is None:
            return f"[mock reply #{self.calls}] re: {last[:60]}"
        props = schema.get("properties", {})
        if "prompts" in props:
            n = _requested_count(last)
            return json.dumps({"prompts": [f"mock prompt {self.calls}-{i} about {last[-30:]}" for i in range(n)]})
        if "message" in props:
            return json.dumps({"message": f"mock follow-up #{self.calls}"})
        out: dict[str, Any] = {}
        for k, v in props.items():
            out[k] = self.rng.randint(3, 5) if v.get("type") == "integer" else f"mock {k}"
        return json.dumps(out)


def _requested_count(text: str) -> int:
    import re

    m = re.search(r"Write (\d+)", text)
    return int(m.group(1)) if m else 5


# --------------------------------------------------------------------------- factory
def make_teacher(cfg: TeacherConfig | dict[str, Any]) -> Teacher:
    if isinstance(cfg, dict):
        cfg = TeacherConfig(**cfg)
    if cfg.backend == "anthropic":
        return AnthropicTeacher(cfg)
    if cfg.backend == "openai":
        return OpenAICompatTeacher(cfg)
    if cfg.backend == "mock":
        return MockTeacher(cfg)
    raise ValueError(f"unknown teacher backend: {cfg.backend}")


def complete_json(teacher: Teacher, system: str, messages: list[Message], schema: dict, **kw) -> dict:
    """Call with a JSON schema and parse; retry once on malformed JSON."""
    for attempt in range(2):
        text = teacher.complete(system, messages, schema=schema, **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt:
                raise TeacherError(f"invalid JSON from teacher: {text[:200]}") from None
            time.sleep(1)
    raise AssertionError("unreachable")
