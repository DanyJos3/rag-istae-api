"""Biblioteca normativa: las normas cargadas, comunes a todas las arquitecturas.

Responde lo que no depende de cómo se genera la respuesta: la lista de
documentos, el texto completo de cada artículo y los datos para verificar citas.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.normativa.corpus import CorpusDocument


class Library:
    def __init__(self, documents: Sequence[CorpusDocument]) -> None:
        self.documents = sorted(documents, key=lambda doc: doc.info.document_id)
        # unit_id → (document_id, texto): lo que usa la verificación de citas.
        self.unit_texts: dict[str, tuple[str, str]] = {
            unit.unit_id: (doc.info.document_id, unit.text) for doc in self.documents for unit in doc.parsed.units
        }
        self.document_names = {doc.info.document_id: doc.info.name for doc in self.documents}
        self._units = {
            unit.unit_id: {
                "unit_id": unit.unit_id,
                "document_id": doc.info.document_id,
                "document_name": doc.info.name,
                "label": unit.label,
                "title": unit.title,
                "text": unit.text,
            }
            for doc in self.documents
            for unit in doc.parsed.units
        }

    def document_summaries(self) -> list[dict[str, object]]:
        """Normas con su ficha del catálogo y la estructura detectada (pantalla Documentos)."""
        return [
            {
                "document_id": doc.info.document_id,
                "name": doc.info.name,
                "doc_type": doc.info.doc_type,
                "issuer": doc.info.issuer,
                "domain": doc.info.domain,
                "rank": doc.info.rank,
                "rank_label": doc.info.rank_label,
                "source_file": doc.info.source_file,
                "articles": sum(1 for unit in doc.parsed.units if unit.kind == "articulo"),
                "units": len(doc.parsed.units),
            }
            for doc in self.documents
        ]

    def get_unit(self, unit_id: str) -> dict[str, str] | None:
        """Texto completo de un artículo o sección, para mostrarlo al usuario."""
        return self._units.get(unit_id)
