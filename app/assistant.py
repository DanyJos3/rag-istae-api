"""Asistente de consulta normativa con contexto largo (variantes A y A+J).

Se construye una vez al iniciar la API y atiende cada pregunta enviando el
corpus completo al modelo. No guarda estado entre preguntas: el historial de
la conversación lo envía el frontend.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.llm.base import ChatMessage, LLMClient, LLMRequest, LLMResult
from app.normativa.citations import parse_answer, sources_by_document
from app.normativa.corpus import Catalog, CorpusDocument, Variant, render_corpus


@dataclass
class Answer:
    """Respuesta final: texto sin la sección «Fuentes» del modelo y fuentes verificadas."""

    body: str
    sources: list[dict[str, Any]]
    abstained: bool
    unknown_citations: list[str]
    cited_without_source: list[str]
    external_sources: list[str]
    verified_quote_ratio: float | None
    variant: str
    corpus_version: str
    model: str
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    finish_reason: str | None
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_prompts(prompts_dir: Path, variant: Variant) -> tuple[str, str]:
    """Instrucciones del sistema (base + jerarquía en A+J) y recordatorio final."""
    prompts_dir = Path(prompts_dir)
    parts = [(prompts_dir / "base_es.txt").read_text(encoding="utf-8").strip()]
    if variant == "AJ":
        parts.append((prompts_dir / "jerarquia_es.txt").read_text(encoding="utf-8").strip())
    reminder = (prompts_dir / "recordatorio_es.txt").read_text(encoding="utf-8").strip()
    return "\n\n".join(parts), reminder


class Assistant:
    def __init__(
        self,
        documents: Sequence[CorpusDocument],
        catalog: Catalog,
        variant: Variant,
        client: LLMClient,
        prompts_dir: Path,
        max_history_messages: int = 6,
        max_output_tokens: int = 8192,
    ) -> None:
        self.variant: Variant = variant
        self.client = client
        self.max_history_messages = max_history_messages
        self.max_output_tokens = max_output_tokens
        self.system, self.reminder = load_prompts(prompts_dir, variant)
        self.context = render_corpus(list(documents), variant, catalog)
        self.document_count = len(documents)
        self.units: dict[str, tuple[str, str]] = {
            unit.unit_id: (doc.info.document_id, unit.text) for doc in documents for unit in doc.parsed.units
        }
        self.document_names = {doc.info.document_id: doc.info.name for doc in documents}
        self._unit_details = {
            unit.unit_id: {
                "unit_id": unit.unit_id,
                "document_id": doc.info.document_id,
                "document_name": doc.info.name,
                "label": unit.label,
                "title": unit.title,
                "text": unit.text,
            }
            for doc in documents
            for unit in doc.parsed.units
        }
        digest = hashlib.sha256()
        for part in (variant, self.system, self.context, self.reminder, client.model):
            digest.update(part.encode("utf-8"))
        # Identifica qué instrucciones, corpus y modelo produjeron cada respuesta.
        self.corpus_version = digest.hexdigest()[:16]

    def get_unit(self, unit_id: str) -> dict[str, str] | None:
        """Texto completo de un artículo o sección, para mostrarlo al usuario."""
        return self._unit_details.get(unit_id)

    def build_request(self, question: str, history: Sequence[ChatMessage] = ()) -> LLMRequest:
        recent = tuple(history)[-self.max_history_messages:] if self.max_history_messages else ()
        return LLMRequest(
            system=self.system,
            context=self.context,
            question=question.strip(),
            history=recent,
            reminder=self.reminder,
            max_output_tokens=self.max_output_tokens,
        )

    def stream(self, question: str, history: Sequence[ChatMessage] = ()) -> Iterator[str | Answer]:
        """Entrega fragmentos de texto y, al final, el ``Answer`` con las citas verificadas."""
        request = self.build_request(question, history)
        started = time.perf_counter()
        pieces: list[str] = []
        result = LLMResult(model=self.client.model)
        for item in self.client.stream(request):
            if isinstance(item, LLMResult):
                result = item
            else:
                pieces.append(item)
                yield item
        yield self._answer("".join(pieces), result, (time.perf_counter() - started) * 1000)

    def _answer(self, text: str, result: LLMResult, latency_ms: float) -> Answer:
        parsed = parse_answer(text, self.units)
        return Answer(
            body=parsed.body,
            sources=sources_by_document(parsed.citations, self.document_names),
            abstained=parsed.abstained,
            unknown_citations=parsed.unknown_ids,
            cited_without_source=parsed.cited_without_source,
            external_sources=parsed.external_sources,
            verified_quote_ratio=parsed.verified_quote_ratio,
            variant=self.variant,
            corpus_version=self.corpus_version,
            model=result.model,
            input_tokens=result.input_tokens,
            cached_input_tokens=result.cached_input_tokens,
            output_tokens=result.output_tokens,
            reasoning_tokens=result.reasoning_tokens,
            finish_reason=result.finish_reason,
            latency_ms=round(latency_ms),
        )
