"""API de consulta normativa del ISTAE.

Endpoints:

- ``GET /health``: estado del servicio y versión del corpus.
- ``GET /units/{unit_id}``: texto completo de un artículo o sección citado.
- ``POST /chat``: responde una pregunta como flujo de eventos (Server-Sent Events):
    ``delta``  → fragmento de texto a medida que el modelo responde;
    ``answer`` → respuesta final con el texto sin la sección «Fuentes» del
                 modelo y las fuentes verificadas, agrupadas por documento;
    ``error``  → mensaje legible si el proveedor falla.

Ejecución: ``uvicorn app.main:create_app --factory``
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.assistant import Answer, Assistant
from app.config import Settings
from app.llm.base import ChatMessage
from app.llm.openai_compatible import LLMError, OpenAICompatibleClient
from app.normativa.corpus import load_catalog, load_corpus
from app.schemas import ChatRequest

logger = logging.getLogger("app")


def build_assistant(settings: Settings) -> Assistant:
    catalog = load_catalog(settings.data_dir / "catalogo_normativo.json")
    documents = load_corpus(settings.data_dir, catalog)
    warnings = [(doc.info.document_id, w) for doc in documents for w in doc.parsed.warnings]
    for document_id, warning in warnings:
        logger.debug("%s: %s", document_id, warning)
    if warnings:
        logger.info("Análisis de estructura: %d advertencias (detalle en nivel DEBUG)", len(warnings))
    client = OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        fallback_models=settings.llm_fallback_models,
        reasoning_effort=settings.llm_reasoning_effort,
        start_timeout_seconds=settings.llm_start_timeout_seconds,
    )
    return Assistant(
        documents,
        catalog,
        settings.variant,
        client,
        settings.prompts_dir,
        max_history_messages=settings.max_history_messages,
        max_output_tokens=settings.llm_max_output_tokens,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error_message(exc: LLMError) -> str:
    if exc.status_code == 429:
        return "Se alcanzó el límite de consultas del modelo. Intente nuevamente en unos minutos."
    if exc.status_code in (408, 502, 503, 504):
        return "El modelo está saturado o tardó demasiado en responder. Intente nuevamente en unos minutos."
    return "El servicio del modelo no respondió correctamente. Intente nuevamente."


def create_app(settings: Settings | None = None, assistant: Assistant | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.assistant = assistant or build_assistant(settings)
        logger.info(
            "Corpus listo: %d documentos, %d unidades, variante %s, versión %s",
            app.state.assistant.document_count,
            len(app.state.assistant.units),
            app.state.assistant.variant,
            app.state.assistant.corpus_version,
        )
        yield
        close = getattr(app.state.assistant.client, "close", None)
        if callable(close):
            close()

    app = FastAPI(title="API de consulta normativa ISTAE", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health(request: Request) -> dict:
        current: Assistant = request.app.state.assistant
        return {
            "status": "ok",
            "variant": current.variant,
            "model": current.client.model,
            "documents": current.document_count,
            "units": len(current.units),
            "corpus_version": current.corpus_version,
        }

    @app.get("/units/{unit_id}")
    def get_unit(unit_id: str, request: Request) -> dict:
        unit = request.app.state.assistant.get_unit(unit_id)
        if unit is None:
            raise HTTPException(status_code=404, detail="El artículo no existe en la normativa disponible.")
        return unit

    @app.post("/chat")
    def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        current: Assistant = request.app.state.assistant
        history = [ChatMessage(turn.role, turn.content) for turn in body.history]

        def events() -> Iterator[str]:
            try:
                for item in current.stream(body.question, history):
                    if isinstance(item, Answer):
                        logger.info(
                            "Respuesta: modelo=%s entrada=%s (caché %s) salida=%s latencia=%sms citas_inválidas=%s",
                            item.model, item.input_tokens, item.cached_input_tokens,
                            item.output_tokens, item.latency_ms, item.unknown_citations,
                        )
                        yield _sse("answer", item.to_dict())
                    else:
                        yield _sse("delta", {"text": item})
            except LLMError as exc:
                logger.error("Error del proveedor (%s): %s", exc.status_code, exc)
                yield _sse("error", {"message": _error_message(exc), "status": exc.status_code})
            except Exception:  # noqa: BLE001 - el flujo ya empezó: se informa como evento
                logger.exception("Error inesperado al responder")
                yield _sse("error", {"message": "Ocurrió un error inesperado. Intente nuevamente.", "status": None})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app
