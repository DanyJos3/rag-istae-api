"""Arma el dataset de RAGAS con las respuestas ya generadas para el Gold Standard.

Por cada pregunta junta la respuesta esperada (Gold), la respuesta del sistema
(resultados/respuestas_<variante>.jsonl) y los artículos que recibió el modelo. Esos
artículos no se guardaron completos, así que se vuelven a pedir a /retrieve con
la misma configuración de la variante (arquitectura C, con o sin reranker). Para
comprobar que son los mismos, cada artículo citado en la respuesta debe estar
entre los recuperados.

Requiere rag-istae-query encendido y RAG_API_KEY. Uso, desde rag-istae-api:

    .venv\\Scripts\\python.exe evaluation\\dataset_ragas.py [--variante C_sin_rerank]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

from generar_respuestas import GOLD_STANDARD, RESULTS, VARIANTES, load_questions, read_results

RETRIEVAL = os.environ.get("RETRIEVAL_URL", "http://127.0.0.1:8100")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Arma el dataset de RAGAS de una variante.")
    parser.add_argument("--variante", choices=VARIANTES, default="C")
    variante = parser.parse_args().variante
    questions = load_questions(GOLD_STANDARD)
    answers = read_results(RESULTS / f"respuestas_{variante}.jsonl")
    headers = {"X-API-Key": os.environ["RAG_API_KEY"]}
    rows, problems = [], []

    for question in questions:
        record = answers.get(question["ID"], {})
        if record.get("status") != "ok":
            problems.append(f"{question['ID']}: sin respuesta generada")
            continue
        body = {"question": question["Pregunta"], "hierarchy": True, "rerank": VARIANTES[variante], "question_id": question["ID"]}
        data = httpx.post(f"{RETRIEVAL}/retrieve", json=body, headers=headers, timeout=180).raise_for_status().json()

        retrieved = [unit["unit_id"] for unit in data["units"]]
        cited = [unit["unit_id"] for source in record["answer"]["sources"] for unit in source["units"]]
        missing = [unit_id for unit_id in cited if unit_id not in retrieved]
        if missing:
            problems.append(f"{question['ID']}: citó artículos que no se recuperaron ahora: {missing}")

        rows.append({
            "id": question["ID"],
            "user_input": question["Pregunta"],
            "response": record["answer"]["body"],
            "reference": question["Respuesta esperada"],
            "retrieved_contexts": [f"[{unit['unit_id']}] {unit['text']}" for unit in data["units"]],
            "retrieved_ids": retrieved,
            "cited_ids": cited,
        })
        print(f"{question['ID']}: {len(retrieved)} artículos, {len(cited)} citados, faltan {len(missing)}", flush=True)

    path = RESULTS / f"dataset_ragas_{variante}.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"\n{len(rows)} preguntas en {path}")
    print("Sin diferencias." if not problems else "Revisar:\n" + "\n".join(problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
