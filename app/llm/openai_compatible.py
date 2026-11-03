"""Cliente para APIs compatibles con OpenAI (OpenRouter, NVIDIA NIM, DeepSeek…).

Usa ``POST {base_url}/chat/completions`` con ``stream: true`` (eventos SSE).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from app.llm.base import LLMRequest, LLMResult, question_text, system_text


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 600,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._http = http_client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "Consulta normativa ISTAE"},
            timeout=httpx.Timeout(timeout_seconds, connect=30),
        )

    def payload(self, request: LLMRequest) -> dict[str, Any]:
        messages = [{"role": "system", "content": system_text(request)}]
        messages += [{"role": m.role, "content": m.content} for m in request.history]
        messages.append({"role": "user", "content": question_text(request)})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def stream(self, request: LLMRequest) -> Iterator[str | LLMResult]:
        result = LLMResult(model=self.model)
        with self._http.stream("POST", "/chat/completions", json=self.payload(request)) as response:
            if response.status_code >= 400:
                response.read()
                raise LLMError(_error_message(response), response.status_code)
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue  # comentarios de keep-alive
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    error = event["error"] or {}
                    raise LLMError(str(error.get("message", error)), error.get("code"))
                result.model = event.get("model") or result.model
                for choice in event.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        yield text
                    result.finish_reason = choice.get("finish_reason") or result.finish_reason
                if event.get("usage"):
                    _apply_usage(result, event["usage"])
        yield result

    def close(self) -> None:
        self._http.close()


def _apply_usage(result: LLMResult, usage: dict[str, Any]) -> None:
    result.input_tokens = usage.get("prompt_tokens")
    result.cached_input_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    result.output_tokens = usage.get("completion_tokens")
    result.reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if usage.get("cost") is not None:
        result.extra["cost_usd"] = usage["cost"]


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        message = error.get("message") or response.text
        raw = (error.get("metadata") or {}).get("raw")
        return f"{message} | {raw}" if raw else message
    except ValueError:
        return response.text
