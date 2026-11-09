"""Cliente del proveedor con respuestas HTTP simuladas."""

from __future__ import annotations

import json

import httpx
import pytest

from app.llm.base import LLMRequest, LLMResult
from app.llm.openai_compatible import LLMError, OpenAICompatibleClient

REQUEST = LLMRequest(system="Instrucciones", context="[INT08-art12] texto", question="¿Pregunta?")

ANSWER = (
    ": OPENROUTER PROCESSING\n\n"
    'data: {"model": "m-1", "choices": [{"delta": {"content": "Hola "}}]}\n\n'
    'data: {"choices": [{"delta": {"content": "mundo"}, "finish_reason": "stop"}]}\n\n'
    'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}\n\n'
    "data: [DONE]\n\n"
)
ONLY_KEEP_ALIVE = ": OPENROUTER PROCESSING\n\n" * 3


def make_client(responses: dict[str, httpx.Response], **options) -> tuple[OpenAICompatibleClient, list[str]]:
    """Cliente cuyo servidor simulado responde según el modelo pedido."""
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        asked.append(model)
        return responses[model]

    http = httpx.Client(base_url="https://x.test", transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(base_url="https://x.test", api_key="k", model="principal", http_client=http, **options)
    return client, asked


def test_stream_yields_text_then_usage() -> None:
    client, _ = make_client({"principal": httpx.Response(200, text=ANSWER)})
    items = list(client.stream(REQUEST))
    assert "".join(i for i in items if isinstance(i, str)) == "Hola mundo"
    result = items[-1]
    assert isinstance(result, LLMResult)
    assert (result.model, result.input_tokens, result.finish_reason) == ("m-1", 10, "stop")


def test_stalled_model_raises_timeout_without_fallback() -> None:
    # Solo comentarios de "sigo procesando": sin tiempo máximo se esperaría para siempre.
    client, _ = make_client({"principal": httpx.Response(200, text=ONLY_KEEP_ALIVE)}, start_timeout_seconds=0)
    with pytest.raises(LLMError) as error:
        list(client.stream(REQUEST))
    assert error.value.status_code == 504


def test_fallback_is_used_when_main_model_stalls_or_is_down() -> None:
    for main_response in (httpx.Response(200, text=ONLY_KEEP_ALIVE), httpx.Response(503, json={"error": {"message": "down"}})):
        client, asked = make_client(
            # El respaldo responde sin esperar (sin comentarios de "sigo procesando").
            {"principal": main_response, "respaldo": httpx.Response(200, text=ANSWER.split("\n\n", 1)[1])},
            fallback_models=["respaldo"],
            start_timeout_seconds=0,
        )
        items = list(client.stream(REQUEST))
        assert asked == ["principal", "respaldo"]
        assert "".join(i for i in items if isinstance(i, str)) == "Hola mundo"


def client_with_keys(exhausted: set[str], **options) -> tuple[OpenAICompatibleClient, list[str]]:
    """Servidor simulado que responde 429 a las claves sin cupo."""
    used: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["Authorization"].removeprefix("Bearer ")
        used.append(key)
        if key in exhausted:
            return httpx.Response(429, json={"error": {"message": "sin cupo"}})
        return httpx.Response(200, text=ANSWER)

    http = httpx.Client(base_url="https://x.test", transport=httpx.MockTransport(handler))
    return OpenAICompatibleClient(base_url="https://x.test", api_key="principal", model="m", http_client=http, **options), used


def test_backup_key_takes_over_and_stays_when_quota_runs_out() -> None:
    client, used = client_with_keys({"principal"}, backup_api_key="respaldo")
    assert "".join(i for i in client.stream(REQUEST) if isinstance(i, str)) == "Hola mundo"
    assert used == ["principal", "respaldo"]
    list(client.stream(REQUEST))
    assert used == ["principal", "respaldo", "respaldo"]


def test_quota_error_is_raised_when_every_key_is_exhausted() -> None:
    client, used = client_with_keys({"principal", "respaldo"}, backup_api_key="respaldo")
    with pytest.raises(LLMError) as error:
        list(client.stream(REQUEST))
    assert error.value.status_code == 429
    assert used == ["principal", "respaldo"]


def test_quota_error_is_raised_without_backup_key() -> None:
    client, used = client_with_keys({"principal"})
    with pytest.raises(LLMError):
        list(client.stream(REQUEST))
    assert used == ["principal"]


def test_invalid_request_is_not_retried() -> None:
    client, asked = make_client(
        {"principal": httpx.Response(400, json={"error": {"message": "bad"}})}, fallback_models=["respaldo"]
    )
    with pytest.raises(LLMError):
        list(client.stream(REQUEST))
    assert asked == ["principal"]
