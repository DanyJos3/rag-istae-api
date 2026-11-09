"""Arquitecturas RAG (B y C) con el servicio de recuperación y el modelo simulados."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.architectures.rag import RagAssistant, RetrievalClient, RetrievalError
from app.config import ROOT, Settings
from app.llm.base import ChatMessage, LLMRequest, LLMResult, system_text
from app.llm.openai_compatible import OpenAICompatibleClient
from app.main import create_app

UNIT_TEXT = "**Artículo 22. Experiencia laboral.** Las horas de experiencia laboral se pueden convalidar."
RETRIEVED = {
    "contract_version": "1.0",
    "run_id": "r1",
    "baseline": "LB-1",
    "mode": {"rerank": True, "hierarchy": True, "architecture": "C"},
    "units": [
        {
            "unit_id": "INT15-art22", "document_id": "INT15", "document_name": "Reglamento de Prácticas Preprofesionales",
            "rank": 6, "label": "Art. 22", "title": "Experiencia laboral", "text": UNIT_TEXT, "partial": False,
            "score": 0.9, "origin": "base", "matched_chunk_ids": ["INT15:00001"],
        },
        {
            "unit_id": "EXT02-art43", "document_id": "EXT02", "document_name": "Reglamento de Régimen Académico",
            "rank": 3, "label": "Art. 43", "title": "", "text": "Texto del artículo 43 […]", "partial": True,
            "score": None, "origin": "hierarchy", "matched_chunk_ids": [],
        },
    ],
    "context": f"=== [INT15] Reglamento de Prácticas Preprofesionales ===\n[INT15-art22] {UNIT_TEXT}",
    "tokens_estimated": 30,
    "warnings": [],
    "cached": False,
    "index_version": {"collection": "c", "units_sha256": "u", "catalog_sha256": "k"},
    "timings_ms": {"total": 10},
}
ANSWER = (
    "Se pueden convalidar [INT15-art22]. También aplica otro artículo [INT99-art1].\n\nFuentes:\n"
    "Reglamento de Prácticas Preprofesionales (INT15)\n"
    "- [INT15-art22] «Las horas de experiencia laboral se pueden convalidar»"
)


class FakeRetrieval:
    def __init__(self, error: RetrievalError | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, bool, int]] = []

    def retrieve(self, question: str, *, hierarchy: bool, turn: int) -> dict:
        self.calls.append((question, hierarchy, turn))
        if self.error:
            raise self.error
        return RETRIEVED


class FakeLLM:
    model = "fake-model"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def stream(self, request: LLMRequest) -> Iterator[str | LLMResult]:
        self.requests.append(request)
        yield ANSWER[:20]
        yield ANSWER[20:]
        yield LLMResult(model="fake-model", input_tokens=100, output_tokens=20, finish_reason="stop")


def assistant(variant: str = "C", retrieval: FakeRetrieval | None = None, llm: FakeLLM | None = None) -> RagAssistant:
    return RagAssistant(variant, retrieval or FakeRetrieval(), llm or FakeLLM(), ROOT / "prompts", max_history_messages=2)


def real_retrieval_client(handler: Callable[[httpx.Request], httpx.Response]) -> RetrievalClient:
    """Cliente construido como en producción (con su cabecera), con la red reemplazada."""
    client = RetrievalClient(base_url="http://retrieval.test", api_key="clave")
    client._http._transport = httpx.MockTransport(handler)
    return client


def read_events(response: httpx.Response) -> list[tuple[str, dict]]:
    parsed = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        parsed.append((fields["event"], json.loads(fields["data"])))
    return parsed


def rag_settings(**overrides) -> Settings:
    return Settings(_env_file=None, llm_api_key="test", rag_api_key="test", **overrides)


# --- RetrievalClient -------------------------------------------------------------


def test_retrieval_client_sends_body_and_api_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=RETRIEVED)

    data = real_retrieval_client(handler).retrieve("¿Convalidan?", hierarchy=True, turn=2)
    assert data["baseline"] == "LB-1"
    assert seen[0].url.path == "/retrieve"
    assert seen[0].headers["X-API-Key"] == "clave"
    assert json.loads(seen[0].content) == {"question": "¿Convalidan?", "hierarchy": True, "turn": 2}


@pytest.mark.parametrize("status", [401, 503])
def test_retrieval_http_errors_become_retrieval_error(status: int) -> None:
    client = real_retrieval_client(lambda _: httpx.Response(status, text="fallo"))
    with pytest.raises(RetrievalError) as error:
        client.retrieve("Hola", hierarchy=False, turn=1)
    assert error.value.status_code == status


def test_retrieval_timeout_becomes_retrieval_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("lento", request=request)

    with pytest.raises(RetrievalError) as error:
        real_retrieval_client(handler).retrieve("Hola", hierarchy=False, turn=1)
    assert error.value.status_code == 504


# --- RagAssistant ----------------------------------------------------------------


def test_followup_retrieves_with_first_user_question() -> None:
    retrieval, llm = FakeRetrieval(), FakeLLM()
    history = [ChatMessage("user", "¿Me convalidan?"), ChatMessage("assistant", "Sí.")]
    list(assistant("C", retrieval, llm).stream("¿Y qué documentos necesito?", history))
    assert retrieval.calls == [("¿Me convalidan?", True, 2)]
    assert llm.requests[0].question == "¿Y qué documentos necesito?"


def test_variant_b_has_no_hierarchy_and_c_uses_its_prompt() -> None:
    retrieval, llm = FakeRetrieval(), FakeLLM()
    list(assistant("B", retrieval, llm).stream("Hola"))
    assert retrieval.calls == [("Hola", False, 1)]
    assert "DOCUMENTOS RECUPERADOS" in llm.requests[0].system
    assert "JERARQUÍA NORMATIVA" not in llm.requests[0].system

    c_llm = FakeLLM()
    list(assistant("C", FakeRetrieval(), c_llm).stream("Hola"))
    assert "JERARQUÍA NORMATIVA" in c_llm.requests[0].system
    assert "artículo 425" not in c_llm.requests[0].system


def test_context_goes_inside_documentos() -> None:
    llm = FakeLLM()
    list(assistant(llm=llm).stream("Hola"))
    assert f"<documentos>\n{RETRIEVED['context']}\n</documentos>" in system_text(llm.requests[0])


def test_citations_are_verified_only_against_retrieved_units() -> None:
    answer = list(assistant().stream("Hola"))[-1]
    assert answer.unknown_citations == ["INT99-art1"]
    assert answer.architecture == "C"
    assert answer.sources == [{
        "document_id": "INT15",
        "document_name": "Reglamento de Prácticas Preprofesionales",
        "units": [{
            "unit_id": "INT15-art22",
            "quote": "Las horas de experiencia laboral se pueden convalidar",
            "quote_verified": True,
            "label": "Art. 22",
            "title": "Experiencia laboral",
            "text": UNIT_TEXT,
            "partial": False,
            "origin": "base",
        }],
    }]


# --- API -------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["B", "C"])
def test_chat_streams_rag_answer(variant: str) -> None:
    architectures = {variant: assistant(variant)}
    with TestClient(create_app(rag_settings(), None, architectures)) as client:
        stream = read_events(client.post("/chat", json={"question": "Hola", "architecture": variant}))
    assert "".join(data["text"] for name, data in stream if name == "delta") == ANSWER
    assert stream[-1][0] == "answer"
    assert stream[-1][1]["architecture"] == variant
    assert stream[-1][1]["sources"][0]["units"][0]["text"] == UNIT_TEXT


def test_chat_uses_c_by_default() -> None:
    retrieval = FakeRetrieval()
    with TestClient(create_app(rag_settings(), None, {"C": assistant("C", retrieval)})) as client:
        client.post("/chat", json={"question": "Hola"})
    assert retrieval.calls == [("Hola", True, 1)]


def test_retrieval_failure_is_reported_as_readable_event() -> None:
    architectures = {"C": assistant("C", FakeRetrieval(RetrievalError("caído", 503)))}
    with TestClient(create_app(rag_settings(), None, architectures)) as client:
        stream = read_events(client.post("/chat", json={"question": "Hola", "architecture": "C"}))
    assert stream == [("error", {
        "message": "El servicio de búsqueda de normativa no respondió. Intente nuevamente en unos minutos.",
        "status": 503,
    })]


def test_without_long_context_data_is_not_read(tmp_path: Path) -> None:
    settings = rag_settings(data_dir=tmp_path / "no-existe")
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        architectures = client.get("/architectures").json()
        unit = client.get("/units/INT08-art12")
        documents = client.get("/documents")
    assert (health["documents"], health["units"]) == (None, None)
    assert health["architectures"] == ["B", "C"]
    assert health["retrieval_url"] == "http://127.0.0.1:8100"
    assert {a["id"]: a["available"] for a in architectures} == {"A": False, "AJ": False, "B": True, "C": True}
    assert unit.status_code == 404
    assert documents.status_code == 404


def test_long_context_can_be_enabled(data_dir: Path) -> None:
    with TestClient(create_app(rag_settings(data_dir=data_dir, enable_long_context=True))) as client:
        health = client.get("/health").json()
        unit = client.get("/units/INT08-art12")
    assert health["architectures"] == ["B", "C", "A", "AJ"]
    assert health["documents"] == 2
    assert unit.status_code == 200


def test_rag_with_real_llm_client_over_sse() -> None:
    """De /retrieve al texto final con los clientes reales y la red simulada."""

    def llm_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert RETRIEVED["context"] in body["messages"][0]["content"]
        chunks = [{"choices": [{"delta": {"content": ANSWER}}], "model": "m"}, {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        return httpx.Response(200, text="".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n")

    llm = OpenAICompatibleClient(
        base_url="http://llm.test",
        api_key="k",
        model="m",
        http_client=httpx.Client(base_url="http://llm.test", transport=httpx.MockTransport(llm_handler)),
    )
    retrieval = real_retrieval_client(lambda _: httpx.Response(200, json=RETRIEVED))
    architectures = {"C": RagAssistant("C", retrieval, llm, ROOT / "prompts")}
    with TestClient(create_app(rag_settings(), None, architectures)) as client:
        stream = read_events(client.post("/chat", json={"question": "Hola", "architecture": "C"}))
    assert stream[-1][0] == "answer"
    assert stream[-1][1]["sources"][0]["units"][0]["quote_verified"] is True
