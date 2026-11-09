# rag-istae-api

Backend del sistema conversacional de consulta sobre la normativa institucional
y nacional del Instituto Superior Tecnológico Alberto Enríquez (ISTAE). Trabajo
de titulación de la Maestría en Inteligencia Artificial Aplicada (UIDE), Grupo 3.

Responde con las arquitecturas **RAG**: **B** (híbrido con reranking) y **C**
(además, jerarquía normativa). Los artículos y el contexto los entrega el servicio
`POST /retrieve` de `rag-istae-query`; esta API llama al modelo y verifica las citas
contra los artículos que realmente se le enviaron. Cada conversación trata un solo
tema: la búsqueda usa la primera pregunta y los seguimientos reciben los mismos artículos.

Las arquitecturas de **contexto largo** (A y A+J) quedan como históricas: solo se
cargan con `ENABLE_LONG_CONTEXT=true`.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado, modelo, arquitecturas disponibles y URL del servicio de recuperación |
| GET | `/architectures` | Arquitecturas que se pueden elegir en el chat y si están disponibles |
| POST | `/chat` | Responde una pregunta como flujo de eventos (SSE) |
| GET | `/documents`, `/units/{unit_id}` | Solo con `ENABLE_LONG_CONTEXT=true`; si no, responden 404 |

`POST /chat` recibe:

```json
{
  "question": "¿Cuál es la nota mínima para aprobar el trabajo de integración curricular?",
  "history": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}],
  "architecture": "C"
}
```

y emite eventos:

- `delta`: `{"text": "…"}`, fragmentos de la respuesta a medida que llegan;
- `answer`: respuesta final. `body` es el texto sin la sección «Fuentes» del
  modelo; `sources` agrupa las citas por documento, con la cita textual, si se
  verificó y el artículo enviado al modelo (`label`, `title`, `text`, `partial`,
  `origin`). Incluye también los identificadores inventados, las fuentes externas
  detectadas y el uso de tokens;
- `error`: `{"message": "…", "status": 429}`, mensaje legible si falla el proveedor
  o el servicio de recuperación.

## Modelo y respaldo

El modelo es `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter) y por defecto no
hay modelo de respaldo: todas las respuestas salen del mismo modelo, como exige la
evaluación de la tesis. Para activar uno, defina `LLM_FALLBACK_MODELS` (lista JSON);
se usa si el principal no responde en `LLM_START_TIMEOUT_SECONDS` (150 s) o está caído.
Cada respuesta indica qué modelo contestó.

**Clave de respaldo:** con `LLM_BACKUP_API_KEY`, si la cuenta principal se queda sin
cupo (error 429 o 402), la consulta se repite con esa clave y el mismo modelo, y se
sigue usando hasta reiniciar el servicio.

## Despliegue (Render)

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:create_app --factory --host 0.0.0.0 --port $PORT`
- Variables: `LLM_API_KEY`, `RETRIEVAL_URL`, `RAG_API_KEY` y `CORS_ORIGINS` (lista
  JSON con la URL del frontend)
- `RAG_API_KEY` es la contraseña compartida con el servicio de recuperación: lleva
  el mismo nombre y valor en ambos. La guía `rag-istae-query/docs/guia_integracion_api.md`
  la llama `RETRIEVAL_API_KEY` en esta API; aquí se unificó en `RAG_API_KEY`.
- Python: `.python-version`

## Estructura

```
app/
  main.py            API (FastAPI): /health, /architectures, /chat, /documents y /units
  config.py          configuración por variables de entorno
  schemas.py         validación de la entrada
  architectures/     arquitecturas que responden: base.py (contrato y catálogo),
                     rag.py (B y C), long_context.py (A y A+J, históricas)
  llm/               contrato con el proveedor y cliente compatible con OpenAI
  normativa/         verificación de citas y, para contexto largo, corpus y división en artículos
prompts/             instrucciones del sistema y recordatorio de formato
data/                texto de las normas y catálogo (solo contexto largo)
tests/               pruebas sin red (recuperación y modelo simulados)
```

## Desarrollo local

Requisitos: Python 3.11 y el servicio de recuperación en marcha
(`rag-istae-query/docs/guia_ambiente_local.md`).

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
copy .env.example .env        # y complete LLM_API_KEY y RAG_API_KEY
uvicorn app.main:create_app --factory --reload
```

Pruebas: `pytest`

La documentación interactiva de la API queda en http://localhost:8000/docs.

## Repositorios relacionados

- `rag-istae-web`: interfaz (Next.js)
- `rag-istae-query`: servicio de recuperación `/retrieve` y línea base RAG
- `rag-istae-ingestion`: extracción, fragmentación e indexación de los documentos
