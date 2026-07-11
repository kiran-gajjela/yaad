"""LLM providers. Local-first: Ollama is the default, Anthropic optional.

Deliberately plain `requests` - no SDK dependencies.
"""
from __future__ import annotations

import os

import requests


class LLMError(Exception):
    pass


class BaseLLM:
    name = "base"
    model = "?"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1000) -> str:
        raise NotImplementedError


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, model: str = "llama3.2:3b", host: str | None = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1000) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,  # thinking models (e.g. gemma4) can burn the whole
                              # token budget reasoning and return empty content;
                              # yaad only ever uses the final answer, not the trace.
            "messages": [{"role": "system", "content": system}, *messages],
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
        except requests.exceptions.ConnectionError as e:
            raise LLMError(
                f"Can't reach Ollama at {self.host}. Is it running?\n"
                f"  ollama serve\n  ollama pull {self.model}"
            ) from e
        if r.status_code != 200:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json()["message"]["content"]


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("Set ANTHROPIC_API_KEY to use --provider anthropic")

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1000) -> str:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            },
            timeout=120,
        )
        if r.status_code != 200:
            raise LLMError(f"Anthropic API error {r.status_code}: {r.text[:300]}")
        data = r.json()
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")


def get_llm(provider: str = "ollama", model: str | None = None) -> BaseLLM:
    if provider == "ollama":
        return OllamaLLM(model or "llama3.2:3b")
    if provider == "anthropic":
        return AnthropicLLM(model or "claude-sonnet-4-6")
    raise LLMError(f"unknown provider: {provider!r} (use 'ollama' or 'anthropic')")
