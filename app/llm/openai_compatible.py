"""Cliente para APIs compatibles con OpenAI (OpenRouter, NVIDIA NIM, DeepSeek…).

Usa ``POST {base_url}/chat/completions`` con ``stream: true`` (eventos SSE).

Modelo de respaldo: si el modelo principal falla o no empieza a responder a
tiempo, se repite la consulta con el siguiente modelo de la lista. Solo se
cambia de modelo antes de mostrar texto, para no mezclar dos respuestas.

Clave de respaldo: si la cuenta se queda sin cupo, se repite la consulta con la
otra clave y el mismo modelo, y se sigue usando hasta reiniciar el servicio.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import httpx

from app.llm.base import LLMRequest, LLMResult, question_text, system_text

logger = logging.getLogger("app")

# Errores en los que tiene sentido probar con el modelo de respaldo.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# Errores de límite o saldo de la cuenta: tiene sentido probar con la clave de respaldo.
QUOTA_STATUS = {402, 429}


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
        fallback_models: list[str] | None = None,
        backup_api_key: str | None = None,
        reasoning_effort: str | None = None,
        start_timeout_seconds: float = 150,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.fallback_models = fallback_models or []
        self._api_keys = [api_key, *([backup_api_key] if backup_api_key else [])]
        self._key_index = 0
        self.reasoning_effort = reasoning_effort
        # Tiempo máximo para que el modelo EMPIECE a responder. Hace falta además del
        # timeout de httpx porque el proveedor envía comentarios de "sigo procesando"
        # que mantienen viva la conexión aunque el modelo esté colgado.
        self.start_timeout_seconds = start_timeout_seconds
        self._http = http_client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-Title": "Consulta normativa ISTAE"},
            # read: tiempo máximo sin recibir nada una vez abierta la conexión.
            timeout=httpx.Timeout(connect=30, read=90, write=60, pool=30),
        )

    def payload(self, request: LLMRequest, model: str) -> dict[str, Any]:
        messages = [{"role": "system", "content": system_text(request)}]
        messages += [{"role": m.role, "content": m.content} for m in request.history]
        messages.append({"role": "user", "content": question_text(request)})
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def stream(self, request: LLMRequest) -> Iterator[str | LLMResult]:
        """Entrega el texto por partes; si el modelo falla antes de escribir, prueba el siguiente."""
        models = [self.model, *self.fallback_models]
        for attempt, model in enumerate(models, start=1):
            wrote_text = False
            try:
                for item in self._stream_with_keys(request, model):
                    wrote_text = wrote_text or isinstance(item, str)
                    yield item
                return
            except LLMError as error:
                can_retry = attempt < len(models) and not wrote_text and error.status_code in RETRYABLE_STATUS
                if not can_retry:
                    raise
                logger.warning("Modelo %s no disponible (%s). Se usa el respaldo %s.", model, error, models[attempt])

    def _stream_with_keys(self, request: LLMRequest, model: str) -> Iterator[str | LLMResult]:
        """Usa la clave vigente; si la cuenta se quedó sin cupo antes de escribir, pasa a la de respaldo."""
        while True:
            key_index = self._key_index
            wrote_text = False
            try:
                for item in self._stream_model(request, model, self._api_keys[key_index]):
                    wrote_text = wrote_text or isinstance(item, str)
                    yield item
                return
            except LLMError as error:
                can_switch = error.status_code in QUOTA_STATUS and not wrote_text and key_index + 1 < len(self._api_keys)
                if not can_switch:
                    raise
                self._key_index = max(self._key_index, key_index + 1)
                logger.warning("La cuenta llegó a su límite (%s). Se usa la clave de respaldo.", error)

    def _stream_model(self, request: LLMRequest, model: str, api_key: str) -> Iterator[str | LLMResult]:
        result = LLMResult(model=model)
        start_deadline = time.monotonic() + self.start_timeout_seconds
        started = False
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            with self._http.stream("POST", "/chat/completions", json=self.payload(request, model), headers=headers) as response:
                if response.status_code >= 400:
                    response.read()
                    raise LLMError(_error_message(response), response.status_code)
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        # Comentario de "sigo procesando": se tolera solo hasta el tiempo máximo.
                        if not started and time.monotonic() >= start_deadline:
                            raise LLMError(f"{model} no empezó a responder en {self.start_timeout_seconds:.0f} s", 504)
                        continue
                    started = True
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
        except httpx.TimeoutException as error:
            raise LLMError(f"{model}: tiempo de espera agotado ({type(error).__name__})", 504) from error
        except httpx.TransportError as error:
            raise LLMError(f"{model}: error de conexión ({type(error).__name__})", 503) from error
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
