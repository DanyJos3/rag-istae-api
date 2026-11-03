"""Limpieza conservadora del Markdown que produce la extracción (Docling).

Solo elimina ruido sin contenido normativo:

- marcadores de imagen vacíos (``<!-- image -->``);
- repeticiones de líneas de membrete de página (UBICACIÓN:, VERSIÓN:, CÓDIGO:,
  ÁREA:, FECHA:, PÁGINA); se conserva la primera aparición de cada una;
- líneas que solo contienen un número de página;
- espacios al final de línea y bloques de líneas vacías.
"""

from __future__ import annotations

import re

_IMAGE = re.compile(r"^\s*<!--\s*image\s*-->\s*$", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_LETTERHEAD = re.compile(
    r"^\s*(?:\*\*)?\s*(?:UBICACI[ÓO]N|VERSI[ÓO]N|C[ÓO]DIGO|[ÁA]REA|FECHA|P[ÁA]GINA)\s*(?:\*\*)?\s*:",
    re.IGNORECASE,
)
_MAX_LETTERHEAD_LENGTH = 100


def clean_markdown(text: str) -> str:
    seen_letterhead: set[str] = set()
    kept: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.rstrip()
        if _IMAGE.match(line) or _PAGE_NUMBER.match(line):
            continue
        stripped = line.strip()
        if len(stripped) <= _MAX_LETTERHEAD_LENGTH and _LETTERHEAD.match(stripped):
            key = stripped.casefold()
            if key in seen_letterhead:
                continue
            seen_letterhead.add(key)
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
