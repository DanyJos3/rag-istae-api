# rag-istae-api

Backend del sistema conversacional de consulta sobre la normativa institucional
y nacional del Instituto Superior Tecnológico Alberto Enríquez (ISTAE). Trabajo
de titulación de la Maestría en Inteligencia Artificial Aplicada (UIDE), Grupo 3.

Hoy implementa la arquitectura de **contexto largo**: el corpus completo
(46 normas, unos 830.000 tokens) se envía al modelo en cada
consulta, y el sistema verifica las citas de la respuesta contra el texto de las
normas. La API está pensada para que otra arquitectura (por ejemplo, RAG) pueda
reemplazar al asistente sin cambiar el contrato con el frontend.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado, modelo, normas cargadas y arquitecturas disponibles |
| GET | `/architectures` | Arquitecturas que se pueden elegir en el chat y si están disponibles |
| GET | `/documents` | Normas del sistema: nombre, tipo, emisor, jerarquía y artículos detectados |
| GET | `/units/{unit_id}` | Texto completo de un artículo o sección citado (para "Ver artículo") |
| POST | `/chat` | Responde una pregunta como flujo de eventos (SSE) |

`POST /chat` recibe:

```json
{
  "question": "¿Cuál es la nota mínima para aprobar el trabajo de integración curricular?",
  "history": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}],
  "architecture": "A"
}
```

y emite eventos:

- `delta`: `{"text": "…"}`, fragmentos de la respuesta a medida que llegan;
- `answer`: respuesta final. `body` es el texto sin la sección «Fuentes» del
  modelo; `sources` agrupa las citas por documento con su **nombre oficial del
  catálogo**, la cita textual y si se verificó contra la norma. Incluye también
  los identificadores inventados, las fuentes externas detectadas y el uso de tokens;
- `error`: `{"message": "…", "status": 429}`, mensaje legible si el proveedor falla.

## Modelo y respaldo

El modelo principal es `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter, ventana
de 1 millón de tokens). Si no empieza a responder en `LLM_START_TIMEOUT_SECONDS`
(150 s) o devuelve un error de caída o saturación, la consulta se repite con los
modelos de `LLM_FALLBACK_MODELS` (por defecto `nvidia/nemotron-3.5-super-vl:free`).
Solo se cambia de modelo antes de mostrar texto, y cada respuesta indica qué
modelo contestó.

Para la evaluación de la tesis, `LLM_FALLBACK_MODELS=[]`: todas las variantes
deben usar el mismo modelo.

## Despliegue (Render)

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:create_app --factory --host 0.0.0.0 --port $PORT`
- Variables: `LLM_API_KEY` y `CORS_ORIGINS` (lista JSON con la URL del frontend)
- Python: `.python-version`

## Estructura

```
app/
  main.py            API (FastAPI): /health, /architectures, /documents, /units y /chat
  config.py          configuración por variables de entorno
  schemas.py         validación de la entrada
  architectures/     arquitecturas que responden: base.py (contrato y catálogo),
                     long_context.py (contexto largo, variantes A y A+J)
  llm/               contrato con el proveedor y cliente compatible con OpenAI
  normativa/         limpieza, división en artículos, corpus, biblioteca de normas y verificación de citas
prompts/             instrucciones del sistema y recordatorio de formato
data/                texto extraído de las normas y catálogo normativo
tests/               pruebas sin red (modelo simulado)
```

`data/` tiene el mismo formato que produce la ingesta (`rag-istae-ingestion`):
`document_mapping.jsonl`, `extracted_text/<sha256>.txt` y `catalogo_normativo.json`.

## Desarrollo local

Requisitos: Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
copy .env.example .env        # y complete LLM_API_KEY
uvicorn app.main:create_app --factory --reload
```

Pruebas: `pytest`

La documentación interactiva de la API queda en http://localhost:8000/docs.

## Repositorios relacionados

- `rag-istae-web`: interfaz (Next.js)
- `rag-istae-ingestion`: extracción y preparación de los documentos
- `rag-istae-query`: arquitecturas evaluadas y scripts de evaluación
