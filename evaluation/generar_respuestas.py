"""Hace las preguntas del Gold Standard al sistema y guarda las respuestas en un Excel.

Cada pregunta se envía a la API como una conversación nueva (arquitectura C). El Excel
deja la respuesta y las fuentes esperadas junto a lo que respondió el sistema, para que
una persona las califique. El script no califica nada.

Requiere rag-istae-query y rag-istae-api encendidos, la API con CHAT_LIMIT_PER_DAY=0
(si no, el límite diario corta la corrida). Uso, desde rag-istae-api:

    .venv\\Scripts\\python.exe evaluation\\generar_respuestas.py

Variante sin reranker: levantar otra API con RETRIEVAL_RERANK=false (por ejemplo en el
puerto 8001) y ejecutar con --variante C_sin_rerank --api http://127.0.0.1:8001.

Se puede interrumpir y volver a ejecutar: continúa con las preguntas que faltan.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill

API = "http://127.0.0.1:8000"
HERE = Path(__file__).parent
GOLD_STANDARD = HERE / "Gold_Standard_Bateria_Pruebas_TFM_v4.xlsx"
RESULTS = HERE / "resultados"
RATE_LIMITED = 429
# Variante evaluada → ¿con reranker? La API debe estar configurada igual (RETRIEVAL_RERANK).
VARIANTES = {"C": True, "C_sin_rerank": False}

HEADERS = [
    "ID", "Origen", "Dominio", "Tipo de consulta", "Estrato", "Tipo de jerarquía",
    "Pregunta", "Respuesta esperada", "Respuesta generada",
    "Fuente esperada 1", "Artículo/sección 1", "Fuente esperada 2", "Artículo/sección 2",
    "Documentos citados", "Artículos o secciones citados",
    "Norma de mayor rango esperada", "Fuentes complementarias (no cuentan para Recall)", "Observaciones",
    "Exactitud (0-2)", "Trazabilidad (0-2)", "¿Cita la norma superior? (Sí/No)", "Evaluador", "Observaciones del evaluador",
    "Tiempo de respuesta (s)", "Tokens de entrada", "Tokens de salida", "Modelo", "Estado",
]
TO_FILL = HEADERS[18:23]
# Encabezados: azul = Gold Standard, verde = respondió el sistema, amarillo = llena el experto, gris = datos de ejecución.
GOLD_COLOR = "9DC3E6"
HEADER_COLORS = {
    **dict.fromkeys(["Respuesta generada", "Documentos citados", "Artículos o secciones citados"], "A9D18E"),
    **dict.fromkeys(TO_FILL, "FFD966"),
    **dict.fromkeys(HEADERS[23:], "D9D9D9"),
}
WIDTHS = {"Pregunta": 38, "Respuesta esperada": 60, "Respuesta generada": 70, "Documentos citados": 40}


def load_questions(path: Path) -> list[dict[str, str]]:
    rows = load_workbook(path, read_only=True, data_only=True)["Gold Standard"].iter_rows(values_only=True)
    header = [str(cell or "").strip() for cell in next(rows)]
    items = [{name: str(value or "").strip() for name, value in zip(header, row)} for row in rows]
    return [item for item in items if item.get("ID")]


def ask(client: httpx.Client, question: str) -> dict[str, Any]:
    """Envía una pregunta y devuelve el evento final de la API."""
    body = {"question": question, "history": [], "architecture": "C"}
    event, final = None, None
    started = time.perf_counter()
    try:
        with client.stream("POST", "/chat", json=body) as response:
            if response.status_code != 200:
                return {"status": "error", "error": f"HTTP {response.status_code}", "http_status": response.status_code}
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: ") and event != "delta":
                    final = (event, json.loads(line[6:]))
    except httpx.HTTPError as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    seconds = round(time.perf_counter() - started, 1)
    if final is None:
        return {"status": "error", "error": "La respuesta se interrumpió", "seconds": seconds}
    name, data = final
    if name != "answer":
        return {"status": "error", "error": data.get("message"), "http_status": data.get("status"), "seconds": seconds}
    return {"status": "ok" if data.get("body") else "vacía", "seconds": seconds, "answer": data}


def plain_text(text: str) -> str:
    """Quita el formato Markdown, que Excel mostraría como asteriscos."""
    text = re.sub(r"^\s*\*\s+", "- ", text, flags=re.MULTILINE)
    return re.sub(r"\*+", "", text)


def read_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {record["id"]: record for record in records}


def row_values(question: dict[str, str], record: dict[str, Any]) -> dict[str, Any]:
    answer = record.get("answer") or {}
    sources = answer.get("sources", [])
    status = record.get("status", "pendiente")
    return {
        **question,
        "Respuesta generada": plain_text(answer.get("body", "")),
        "Documentos citados": "\n".join(f"{d['document_id']} – {d['document_name']}" for d in sources),
        "Artículos o secciones citados": "\n".join(
            [f"{d['document_id']} · {u.get('label') or u['unit_id']}" for d in sources for u in d["units"]]
            + [f"{citation} (no reconocida por el sistema)" for citation in answer.get("unknown_citations", [])]
        ),
        "¿Cita la norma superior? (Sí/No)": "N/A" if question.get("Estrato") == "Directa" else "",
        "Tiempo de respuesta (s)": record.get("seconds", ""),
        "Tokens de entrada": answer.get("input_tokens", ""),
        "Tokens de salida": answer.get("output_tokens", ""),
        "Modelo": answer.get("model", ""),
        "Estado": status if status != "error" else f"error: {record.get('error')}",
    }


def write_excel(path: Path, questions: list[dict[str, str]], results: dict[str, dict[str, Any]]) -> None:
    sheet = (workbook := Workbook()).active
    sheet.title = "Respuestas"
    sheet.append(HEADERS)
    for question in questions:
        values = row_values(question, results.get(question["ID"], {}))
        row = [values.get(header, "") for header in HEADERS]
        sheet.append([ILLEGAL_CHARACTERS_RE.sub("", value) if isinstance(value, str) else value for value in row])
    for column, header in zip(sheet.iter_cols(), HEADERS):
        sheet.column_dimensions[column[0].column_letter].width = WIDTHS.get(header, 20)
        column[0].font = Font(bold=True)
        column[0].fill = PatternFill("solid", fgColor=HEADER_COLORS.get(header, GOLD_COLOR))
        for cell in column:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if isinstance(cell.value, str):
                cell.data_type = "s"  # un texto que empiece con "=" no debe tomarse como fórmula
    sheet.freeze_panes = "B2"
    sheet.auto_filter.ref = sheet.dimensions
    workbook.save(path)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Genera las respuestas del sistema para el Gold Standard.")
    parser.add_argument("--preguntas", help="IDs separados por coma, por ejemplo Q01,Q27. Por defecto, todas.")
    parser.add_argument("--variante", choices=VARIANTES, default="C", help="Nombre de los archivos de resultados.")
    parser.add_argument("--api", default=API, help=f"URL de la API (por defecto {API}).")
    args = parser.parse_args()

    questions = load_questions(GOLD_STANDARD)
    if args.preguntas:
        wanted = {item.strip() for item in args.preguntas.split(",")}
        questions = [question for question in questions if question["ID"] in wanted]
    RESULTS.mkdir(exist_ok=True)
    raw_path, excel_path = RESULTS / f"respuestas_{args.variante}.jsonl", RESULTS / f"respuestas_{args.variante}.xlsx"
    results = read_results(raw_path)

    timeout = httpx.Timeout(connect=10, read=400, write=60, pool=10)
    with httpx.Client(base_url=args.api, timeout=timeout) as client:
        for number, question in enumerate(questions, start=1):
            prefix = f"[{number}/{len(questions)}] {question['ID']}"
            if results.get(question["ID"], {}).get("status") == "ok":
                print(f"{prefix}: ya estaba, se omite")
                continue
            record = {"id": question["ID"], "question": question["Pregunta"], **ask(client, question["Pregunta"])}
            results[record["id"]] = record
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            write_excel(excel_path, questions, results)
            print(f"{prefix}: {record['status']} ({record.get('error') or str(record.get('seconds')) + ' s'})", flush=True)
            if record.get("http_status") == RATE_LIMITED:
                print("Se alcanzó el límite del modelo. Vuelva a ejecutar más tarde: seguirá donde quedó.")
                break

    write_excel(excel_path, questions, results)
    done = sum(1 for question in questions if results.get(question["ID"], {}).get("status") == "ok")
    print(f"Listas {done} de {len(questions)}. Resultado: {excel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
