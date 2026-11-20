"""Califica con RAGAS las respuestas del Gold Standard (dataset de dataset_ragas.py).

El juez es GPT-4o-mini vía OpenRouter, un modelo distinto del que generó las
respuestas, con temperatura 0. Guarda cada nota apenas la obtiene, así que se
puede interrumpir y volver a ejecutar: continúa con lo que falta.

Requiere LLM_API_KEY (OpenRouter, con crédito) y el entorno .venv-ragas
(requirements-ragas.txt). Uso:

    .venv-ragas\\Scripts\\python.exe evaluation\\evaluar_ragas.py [--variante C_sin_rerank]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError
from openpyxl import Workbook
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, FactualCorrectness, Faithfulness,
)

from generar_respuestas import RESULTS, VARIANTES
JUDGE = "openai/gpt-4o-mini"
EMBEDDINGS = "openai/text-embedding-3-small"
OPENROUTER = "https://openrouter.ai/api/v1"


def build_metrics() -> dict:
    client = AsyncOpenAI(api_key=os.environ["LLM_API_KEY"], base_url=OPENROUTER)
    llm = llm_factory(JUDGE, client=client, temperature=0, max_tokens=4096)
    embeddings = embedding_factory("openai", model=EMBEDDINGS, client=client)
    # Qué datos usa cada métrica: pregunta, respuesta generada, artículos recuperados, respuesta esperada.
    return {
        "faithfulness": (Faithfulness(llm=llm), ["user_input", "response", "retrieved_contexts"]),
        "factual_correctness": (FactualCorrectness(llm=llm), ["response", "reference"]),
        "context_recall": (ContextRecall(llm=llm), ["user_input", "retrieved_contexts", "reference"]),
        "context_precision": (ContextPrecisionWithReference(llm=llm), ["user_input", "reference", "retrieved_contexts"]),
        "answer_relevancy": (AnswerRelevancy(llm=llm, embeddings=embeddings), ["user_input", "response"]),
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_excel(path: Path, rows: list[dict], scores: dict, metrics: list[str]) -> None:
    sheet = (workbook := Workbook()).active
    sheet.title = "RAGAS"
    sheet.append(["ID", *metrics])
    for row in rows:
        sheet.append([row["id"], *[scores.get((row["id"], metric)) for metric in metrics]])
    sheet.append(["Promedio", *[
        round(sum(values) / len(values), 3) if (values := [v for (_, m), v in scores.items() if m == metric and v is not None]) else None
        for metric in metrics
    ]])
    workbook.save(path)


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Califica con RAGAS una variante.")
    parser.add_argument("--variante", choices=VARIANTES, default="C")
    variante = parser.parse_args().variante
    rows = read_jsonl(RESULTS / f"dataset_ragas_{variante}.jsonl")
    scores_path = RESULTS / f"ragas_{variante}.jsonl"
    scores = {(s["id"], s["metric"]): s["score"] for s in read_jsonl(scores_path) if s.get("score") is not None}
    metrics = build_metrics()

    for row in rows:
        for name, (metric, fields) in metrics.items():
            if (row["id"], name) in scores:
                continue
            try:
                result = await metric.ascore(**{field: row[field] for field in fields})
            except RateLimitError:
                print("Se alcanzó el límite del juez. Vuelva a ejecutar más tarde: seguirá donde quedó.")
                write_excel(RESULTS / f"ragas_{variante}.xlsx", rows, scores, list(metrics))
                return 1
            except Exception as error:  # una métrica que falla no detiene el resto
                print(f"{row['id']} {name}: error {type(error).__name__}: {error}")
                continue
            scores[(row["id"], name)] = result.value
            with scores_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": row["id"], "metric": name, "score": result.value, "judge": JUDGE}) + "\n")
            print(f"{row['id']} {name}: {result.value}", flush=True)

    write_excel(RESULTS / f"ragas_{variante}.xlsx", rows, scores, list(metrics))
    print(f"Listo: {RESULTS / f'ragas_{variante}.xlsx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
