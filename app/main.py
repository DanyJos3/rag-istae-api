"""API de consulta normativa del ISTAE.

Endpoints:

- ``GET /health``: estado del servicio.
- ``GET /architectures``: arquitecturas que se pueden elegir en el chat y si están disponibles.
- ``GET /documents``: normas con las que trabaja el sistema (catálogo y estructura detectada).
- ``GET /units/{unit_id}``: texto completo de un artículo o sección citado.
- ``POST /chat``: responde una pregunta con la arquitectura elegida, como flujo de eventos (SSE):
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

from app.architectures.base import ARCHITECTURES, Answer, ChatArchitecture
from app.architectures.long_context import LongContextAssistant
from app.architectures.rag import RagAssistant, RetrievalClient, RetrievalError
from app.config import Settings
from app.llm.base import ChatMessage
from app.llm.openai_compatible import LLMError, OpenAICompatibleClient
from app.normativa.corpus import load_catalog, load_corpus
from app.normativa.library import Library
from app.schemas import ChatRequest

logger = logging.getLogger("app")


def build_architectures(settings: Settings) -> tuple[Library | None, dict[str, ChatArchitecture]]:
    """Arma las arquitecturas RAG; las normas completas solo se cargan si se pide contexto largo."""
    client = OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        fallback_models=settings.llm_fallback_models,
        reasoning_effort=settings.llm_reasoning_effort,
        start_timeout_seconds=settings.llm_start_timeout_seconds,
    )
    options = {"max_history_messages": settings.max_history_messages, "max_output_tokens": settings.llm_max_output_tokens}
    retrieval = RetrievalClient(
        base_url=settings.retrieval_url,
        api_key=settings.rag_api_key.get_secret_value(),
        timeout_seconds=settings.retrieval_timeout_seconds,
    )
    architectures: dict[str, ChatArchitecture] = {
        "B": RagAssistant("B", retrieval, client, settings.prompts_dir, **options),
        "C": RagAssistant("C", retrieval, client, settings.prompts_dir, **options),
    }

    library = None
    if settings.enable_long_context:  # histórico: carga las 46 normas solo si se pide
        catalog = load_catalog(settings.data_dir / "catalogo_normativo.json")
        documents = load_corpus(settings.data_dir, catalog)
        warnings = [(doc.info.document_id, w) for doc in documents for w in doc.parsed.warnings]
        for document_id, warning in warnings:
            logger.debug("%s: %s", document_id, warning)
        if warnings:
            logger.info("Análisis de estructura: %d advertencias (detalle en nivel DEBUG)", len(warnings))
        library = Library(documents)
        architectures["A"] = LongContextAssistant(library, catalog, "A", client, settings.prompts_dir, **options)
        architectures["AJ"] = LongContextAssistant(library, catalog, "AJ", client, settings.prompts_dir, **options)
    return library, architectures


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error_message(exc: LLMError) -> str:
    if exc.status_code == 429:
        return "Se alcanzó el límite de consultas del modelo. Intente nuevamente en unos minutos."
    if exc.status_code in (408, 502, 503, 504):
        return "El modelo está saturado o tardó demasiado en responder. Intente nuevamente en unos minutos."
    return "El servicio del modelo no respondió correctamente. Intente nuevamente."


def create_app(
    settings: Settings | None = None,
    library: Library | None = None,
    architectures: dict[str, ChatArchitecture] | None = None,
) -> FastAPI:
    """``library`` y ``architectures`` se pasan solo en las pruebas (modelo simulado)."""
    settings = settings or Settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if architectures is not None:
            app.state.library, app.state.architectures = library, architectures
        else:
            app.state.library, app.state.architectures = build_architectures(settings)
        logger.info("Listo: arquitecturas %s", ", ".join(app.state.architectures))
        yield
        for architecture in app.state.architectures.values():
            for attribute in ("client", "retrieval"):
                resource = getattr(architecture, attribute, None)
                if callable(getattr(resource, "close", None)):
                    resource.close()

    app = FastAPI(title="API de consulta normativa ISTAE", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health(request: Request) -> dict:
        current_library = request.app.state.library
        return {
            "status": "ok",
            "model": settings.llm_model,
            "documents": len(current_library.documents) if current_library else None,
            "units": len(current_library.unit_texts) if current_library else None,
            "architectures": list(request.app.state.architectures),
            "retrieval_url": settings.retrieval_url,
        }

    @app.get("/architectures")
    def list_architectures(request: Request) -> list[dict]:
        available = request.app.state.architectures
        return [
            {"id": info.id, "name": info.name, "description": info.description, "available": info.id in available}
            for info in ARCHITECTURES
        ]

    @app.get("/documents")
    def list_documents(request: Request) -> list[dict]:
        if request.app.state.library is None:
            raise HTTPException(status_code=404, detail="La lista de documentos no está disponible.")
        return request.app.state.library.document_summaries()

    @app.get("/units/{unit_id}")
    def get_unit(unit_id: str, request: Request) -> dict:
        current_library = request.app.state.library
        unit = current_library.get_unit(unit_id) if current_library else None
        if unit is None:
            raise HTTPException(status_code=404, detail="El artículo no existe en la normativa disponible.")
        return unit

    @app.post("/chat")
    def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        architecture = request.app.state.architectures.get(body.architecture)
        if architecture is None:
            raise HTTPException(status_code=400, detail=f"La arquitectura {body.architecture} no está disponible.")
        history = [ChatMessage(turn.role, turn.content) for turn in body.history]

        def events() -> Iterator[str]:
            try:
                for item in architecture.stream(body.question, history):
                    if isinstance(item, Answer):
                        logger.info(
                            "Respuesta %s: modelo=%s entrada=%s (caché %s) salida=%s latencia=%sms citas_inválidas=%s",
                            item.architecture, item.model, item.input_tokens, item.cached_input_tokens,
                            item.output_tokens, item.latency_ms, item.unknown_citations,
                        )
                        yield _sse("answer", item.to_dict())
                    else:
                        yield _sse("delta", {"text": item})
            except LLMError as exc:
                logger.error("Error del proveedor (%s): %s", exc.status_code, exc)
                yield _sse("error", {"message": _error_message(exc), "status": exc.status_code})
            except RetrievalError as exc:
                logger.error("Error del servicio de recuperación (%s): %s", exc.status_code, exc)
                yield _sse("error", {
                    "message": "El servicio de búsqueda de normativa no respondió. Intente nuevamente en unos minutos.",
                    "status": exc.status_code,
                })
            except Exception:  # noqa: BLE001 - el flujo ya empezó: se informa como evento
                logger.exception("Error inesperado al responder")
                yield _sse("error", {"message": "Ocurrió un error inesperado. Intente nuevamente.", "status": None})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app
