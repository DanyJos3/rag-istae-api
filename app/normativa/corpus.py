"""Carga del corpus normativo y armado del texto que recibe el modelo.

Entrada (carpeta ``data``), en el mismo formato que produce la ingesta:

- ``document_mapping.jsonl``: una línea por documento (``document_id``,
  ``relative_path``, ``source_sha256``);
- ``extracted_text/<source_sha256>.txt``: el texto extraído de cada documento;
- ``catalogo_normativo.json``: nombre oficial, rango jerárquico y relaciones.

Variantes:

- ``A``: documentos completos en orden por código.
- ``AJ``: los mismos documentos ordenados por rango, con el rango en cada
  encabezado, la escala jerárquica al inicio y las relaciones entre normas al final.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.normativa.cleaning import clean_markdown
from app.normativa.models import DocumentInfo, ParsedDocument, Relation
from app.normativa.structure import parse_document

Variant = Literal["A", "AJ"]
_UNKNOWN_RANK = 99


@dataclass(frozen=True)
class Catalog:
    documents: dict[str, DocumentInfo]
    relations: list[Relation]
    rank_labels: dict[str, str]


@dataclass
class CorpusDocument:
    info: DocumentInfo
    parsed: ParsedDocument


def load_catalog(path: Path) -> Catalog:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return Catalog(
        documents={item["document_id"]: DocumentInfo(**item) for item in data.get("documents", [])},
        relations=[Relation(**item) for item in data.get("relations", [])],
        rank_labels=data.get("rank_labels", {}),
    )


def _fallback_document(document_id: str, source_file: str) -> DocumentInfo:
    """Ficha mínima para un documento que no figura en el catálogo."""
    name = re.sub(r"^\s*(?:INT|EXT)\d{2}\s*-\s*", "", Path(source_file).stem)
    return DocumentInfo(document_id, document_id, name, "", "", "", _UNKNOWN_RANK, "Sin clasificar", source_file)


def load_corpus(data_dir: Path, catalog: Catalog) -> list[CorpusDocument]:
    data_dir = Path(data_dir)
    lines = (data_dir / "document_mapping.jsonl").read_text(encoding="utf-8").splitlines()
    rows = sorted((json.loads(line) for line in lines if line.strip()), key=lambda row: row["document_id"])
    documents = []
    for row in rows:
        text = (data_dir / "extracted_text" / f"{row['source_sha256']}.txt").read_text(encoding="utf-8")
        document_id = row["document_id"]
        info = catalog.documents.get(document_id) or _fallback_document(document_id, row["relative_path"])
        documents.append(CorpusDocument(info, parse_document(document_id, clean_markdown(text))))
    return documents


def _document_header(info: DocumentInfo, variant: Variant) -> str:
    if variant == "AJ":
        return (
            f"=== [{info.document_id}] {info.name} · Jerarquía {info.rank}: {info.rank_label}"
            f" · Emisor: {info.issuer or 'no registrado'} ==="
        )
    return f"=== [{info.document_id}] {info.name} ==="


def _hierarchy_preamble(rank_labels: dict[str, str]) -> str:
    lines = ["=== ESCALA DE JERARQUÍA NORMATIVA (1 = mayor rango) ==="]
    lines += [f"{key}. {label}" for key, label in sorted(rank_labels.items(), key=lambda item: int(item[0]))]
    lines.append("Los documentos aparecen ordenados de mayor a menor jerarquía.")
    return "\n".join(lines)


def _relations_block(relations: list[Relation], names: dict[str, str]) -> str:
    lines = ["=== RELACIONES ENTRE NORMAS ==="]
    for rel in relations:
        source = f"[{rel.source_document_id}] {names.get(rel.source_document_id, rel.source_code)}" if rel.source_document_id else rel.source_code
        target = f"[{rel.target_document_id}] {names.get(rel.target_document_id, rel.target_code)}" if rel.target_document_id else rel.target_code
        lines.append(f"- {source} — {rel.relation} → {target}")
    return "\n".join(lines)


def render_corpus(documents: list[CorpusDocument], variant: Variant, catalog: Catalog) -> str:
    """Texto completo que se envía al modelo como contexto."""
    if variant == "AJ":
        ordered = sorted(documents, key=lambda doc: (doc.info.rank, doc.info.document_id))
    else:
        ordered = sorted(documents, key=lambda doc: doc.info.document_id)
    blocks: list[str] = []
    if variant == "AJ":
        blocks.append(_hierarchy_preamble(catalog.rank_labels))
    for doc in ordered:
        body = "\n\n".join(f"[{unit.unit_id}] {unit.text}" for unit in doc.parsed.units)
        blocks.append(f"{_document_header(doc.info, variant)}\n{body}")
    if variant == "AJ" and catalog.relations:
        names = {doc.info.document_id: doc.info.name for doc in documents}
        blocks.append(_relations_block(catalog.relations, names))
    return "\n\n".join(blocks) + "\n"
