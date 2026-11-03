"""Datos mínimos de prueba: dos normas pequeñas con el formato de la ingesta."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

INT08 = """**Artículo 1.** Ámbito. Regula la titulación.

**Artículo 2.** Objeto. Normar el proceso.

**Artículo 12. Declaración de egresado.** Un estudiante será declarado egresado cuando cumpla con los siguientes requisitos:

- a) Declaratoria de aptitud con la documentación pertinente.

**Artículo 13. Requisitos para titularse.** Haber sido declarado egresado.
"""
EXT01 = "Art. 1.- Ámbito.- Esta Ley regula el sistema.\n\nArt. 2.- Objeto.- Definir principios.\n\nArt. 3.- Fines.- Formar profesionales."


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "extracted_text").mkdir()
    rows = [
        {"document_id": "INT08", "relative_path": "INTERNOS/INT08 - TITULACION.docx", "source_sha256": "aaa"},
        {"document_id": "EXT01", "relative_path": "EXTERNOS/EXT01 - LOES.pdf", "source_sha256": "bbb"},
    ]
    (tmp_path / "document_mapping.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "extracted_text" / "aaa.txt").write_text(INT08, encoding="utf-8")
    (tmp_path / "extracted_text" / "bbb.txt").write_text(EXT01, encoding="utf-8")
    catalog = {
        "rank_labels": {"2": "Ley orgánica", "6": "Reglamento o normativa interna"},
        "documents": [
            {"document_id": "EXT01", "code": "EXT-LOES", "name": "Ley Orgánica de Educación Superior (LOES)",
             "doc_type": "Externo", "issuer": "Asamblea Nacional", "domain": "", "rank": 2,
             "rank_label": "Ley orgánica", "source_file": "EXT01 - LOES.pdf"},
            {"document_id": "INT08", "code": "INT-RTG", "name": "Reglamento de Titulación", "doc_type": "Interno",
             "issuer": "OCS ISTAE", "domain": "", "rank": 6, "rank_label": "Reglamento o normativa interna",
             "source_file": "INT08 - TITULACION.docx"},
        ],
        "relations": [],
    }
    (tmp_path / "catalogo_normativo.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    return tmp_path
