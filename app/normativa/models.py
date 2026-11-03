"""Tipos del corpus normativo: fichas de documentos, relaciones y unidades citables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

UnitKind = Literal["preambulo", "articulo", "seccion", "disposicion"]
StructureMode = Literal["articulado", "secciones"]


@dataclass(frozen=True)
class DocumentInfo:
    """Ficha de una norma, tomada del catálogo normativo."""

    document_id: str
    code: str
    name: str
    doc_type: str
    issuer: str
    domain: str
    rank: int
    rank_label: str
    source_file: str


@dataclass(frozen=True)
class Relation:
    """Relación entre dos normas (por ejemplo, "Desarrolla")."""

    source_document_id: str | None
    source_code: str
    relation: str
    target_document_id: str | None
    target_code: str
    verification: str = ""


@dataclass
class Unit:
    """Unidad citable: artículo, sección, disposición o preámbulo.

    ``unit_id`` es estable y legible (``INT08-art12``): depende de la norma y de
    su numeración, no de la posición del texto ni de la herramienta de extracción.
    """

    unit_id: str
    kind: UnitKind
    label: str
    title: str
    path: str
    text: str


@dataclass
class ParsedDocument:
    """Documento dividido en unidades, con advertencias del análisis."""

    document_id: str
    mode: StructureMode
    units: list[Unit]
    warnings: list[str] = field(default_factory=list)
