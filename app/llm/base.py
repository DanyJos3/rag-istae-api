"""Contrato con el proveedor del modelo, independiente de la marca.

Orden del prompt recomendado para contextos largos:

    1. instrucciones del sistema
    2. documentos completos            ← prefijo estable, reutilizable en caché
    3. historial de la conversación
    4. pregunta actual + recordatorio breve de las reglas de formato
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMRequest:
    system: str
    context: str
    question: str
    history: tuple[ChatMessage, ...] = ()
    reminder: str = ""
    max_output_tokens: int = 8192


@dataclass
class LLMResult:
    """Datos de uso que llegan al terminar la respuesta."""

    model: str
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


class LLMClient(Protocol):
    model: str

    def stream(self, request: LLMRequest) -> Iterator[str | LLMResult]:
        """Entrega fragmentos de texto y, al final, un ``LLMResult``."""


def system_text(request: LLMRequest) -> str:
    return f"{request.system}\n\n<documentos>\n{request.context}\n</documentos>"


def question_text(request: LLMRequest) -> str:
    text = f"<pregunta>\n{request.question.strip()}\n</pregunta>"
    return f"{text}\n\n{request.reminder.strip()}" if request.reminder.strip() else text
