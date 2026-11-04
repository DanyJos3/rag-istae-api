"""API con un modelo simulado (sin red ni cuota)."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi.testclient import TestClient

from app.architectures.long_context import LongContextAssistant
from app.config import ROOT, Settings
from app.llm.base import LLMRequest, LLMResult
from app.llm.openai_compatible import LLMError
from app.main import create_app
from app.normativa.corpus import load_catalog, load_corpus
from app.normativa.library import Library

ANSWER = (
    "Necesitas la declaratoria de aptitud [INT08-art12].\n\nFuentes:\nReglamento de Titulación (INT08)\n"
    "- [INT08-art12] «Declaratoria de aptitud con la documentación pertinente»"
)


class FakeClient:
    model = "fake-model"

    def __init__(self, error: LLMError | None = None) -> None:
        self.error = error
        self.requests: list[LLMRequest] = []

    def stream(self, request: LLMRequest) -> Iterator[str | LLMResult]:
        self.requests.append(request)
        if self.error:
            raise self.error
        yield ANSWER[:30]
        yield ANSWER[30:]
        yield LLMResult(model="fake-model", input_tokens=1000, cached_input_tokens=900, output_tokens=50, finish_reason="stop")


def make_client(data_dir, fake: FakeClient) -> TestClient:
    """API con las dos arquitecturas de contexto largo (A y A+J) usando el modelo simulado."""
    catalog = load_catalog(data_dir / "catalogo_normativo.json")
    library = Library(load_corpus(data_dir, catalog))
    architectures = {
        variant: LongContextAssistant(library, catalog, variant, fake, ROOT / "prompts", max_history_messages=2)
        for variant in ("A", "AJ")
    }
    return TestClient(create_app(Settings(llm_api_key="test"), library, architectures))


def read_events(response) -> list[tuple[str, dict]]:
    parsed = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        parsed.append((fields["event"], json.loads(fields["data"])))
    return parsed


def test_health_reports_corpus(data_dir) -> None:
    with make_client(data_dir, FakeClient()) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 2
    assert body["architectures"] == ["A", "AJ"]


def test_chat_streams_text_then_verified_answer(data_dir) -> None:
    fake = FakeClient()
    with make_client(data_dir, fake) as client:
        history = [{"role": "user", "content": f"p{i}"} for i in range(4)]
        response = client.post("/chat", json={"question": "  ¿Qué necesito para egresar?  ", "history": history})
    stream = read_events(response)
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "".join(data["text"] for name, data in stream if name == "delta") == ANSWER
    name, answer = stream[-1]
    assert name == "answer"
    assert answer["body"] == "Necesitas la declaratoria de aptitud [INT08-art12]."
    assert answer["sources"][0]["document_name"] == "Reglamento de Titulación"
    assert answer["sources"][0]["units"][0] == {
        "unit_id": "INT08-art12",
        "quote": "Declaratoria de aptitud con la documentación pertinente",
        "quote_verified": True,
    }
    assert answer["cached_input_tokens"] == 900
    request = fake.requests[0]
    assert request.question == "¿Qué necesito para egresar?"
    assert [m.content for m in request.history] == ["p2", "p3"]
    assert request.reminder.startswith("Antes de responder")


def test_rate_limit_is_reported_as_readable_event(data_dir) -> None:
    with make_client(data_dir, FakeClient(LLMError("Rate limit", 429))) as client:
        stream = read_events(client.post("/chat", json={"question": "Hola"}))
    assert stream == [("error", {
        "message": "Se alcanzó el límite de consultas del modelo. Intente nuevamente en unos minutos.",
        "status": 429,
    })]


def test_empty_question_is_rejected(data_dir) -> None:
    with make_client(data_dir, FakeClient()) as client:
        assert client.post("/chat", json={"question": "   "}).status_code == 422


def test_unit_returns_full_article_or_404(data_dir) -> None:
    with make_client(data_dir, FakeClient()) as client:
        found = client.get("/units/INT08-art12")
        missing = client.get("/units/INT08-art99")
    assert found.json()["document_name"] == "Reglamento de Titulación"
    assert found.json()["label"] == "Art. 12"
    assert "Declaratoria de aptitud" in found.json()["text"]
    assert missing.status_code == 404


def test_documents_lists_catalog_and_structure(data_dir) -> None:
    with make_client(data_dir, FakeClient()) as client:
        documents = client.get("/documents").json()
    assert [d["document_id"] for d in documents] == ["EXT01", "INT08"]
    int08 = documents[1]
    assert (int08["name"], int08["doc_type"], int08["rank"]) == ("Reglamento de Titulación", "Interno", 6)
    assert int08["articles"] == 4


def test_architectures_mark_rag_as_not_available_yet(data_dir) -> None:
    with make_client(data_dir, FakeClient()) as client:
        architectures = client.get("/architectures").json()
    assert [(a["id"], a["available"]) for a in architectures] == [("A", True), ("AJ", True), ("B", False), ("C", False)]


def test_chat_uses_the_chosen_architecture(data_dir) -> None:
    fake = FakeClient()
    with make_client(data_dir, fake) as client:
        stream = read_events(client.post("/chat", json={"question": "Hola", "architecture": "AJ"}))
        unavailable = client.post("/chat", json={"question": "Hola", "architecture": "B"})
    assert stream[-1][1]["architecture"] == "AJ"
    assert "JERARQUÍA NORMATIVA" in fake.requests[0].system
    assert unavailable.status_code == 400
