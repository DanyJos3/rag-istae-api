"""Lo común a todas las arquitecturas: catálogo, contrato y formato de la respuesta.

Para agregar una arquitectura (por ejemplo, RAG): implementar una clase con el
método ``stream`` de ``ChatArchitecture``, registrarla en ``main.build_architectures``
y marcarla como disponible. El endpoint /chat y el frontend no cambian.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from app.llm.base import ChatMessage


@dataclass(frozen=True)
class ArchitectureInfo:
    id: str
    name: str
    description: str


# Las cuatro arquitecturas del diseño 2×2 de la tesis, en el orden en que se muestran.
ARCHITECTURES = [
    ArchitectureInfo("A", "Contexto largo (histórico)", "El modelo lee las 46 normas completas en cada consulta."),
    ArchitectureInfo(
        "AJ",
        "Contexto largo + jerarquía (histórico)",
        "Las normas completas, ordenadas por rango, con la regla de prevalencia del art. 425 de la Constitución.",
    ),
    ArchitectureInfo(
        "B",
        "RAG híbrido",
        "Busca los artículos más pertinentes (semántica + palabras clave + reordenamiento) y solo envía esos al modelo.",
    ),
    ArchitectureInfo("C", "RAG jerárquico", "Recuperación de artículos que además considera la jerarquía normativa."),
]


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
    architecture: str
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


class ChatArchitecture(Protocol):
    """Contrato de una arquitectura: texto por partes y, al final, el ``Answer``."""

    def stream(self, question: str, history: Sequence[ChatMessage] = ()) -> Iterator[str | Answer]: ...
