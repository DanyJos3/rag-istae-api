"""Esquemas de entrada de la API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(max_length=20000)]


class ChatRequest(BaseModel):
    question: Question
    # El frontend envía toda la conversación: RAG busca con la primera pregunta y al modelo solo van los últimos turnos.
    history: list[ChatTurn] = Field(default_factory=list, max_length=200)
    # Id de la arquitectura elegida en el chat (ver GET /architectures).
    architecture: str = "C"
