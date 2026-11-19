"""Configuración por variables de entorno (o archivo ``.env`` en desarrollo).

La API key solo se lee del entorno; nunca va en el código ni en el repositorio.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    llm_api_key: SecretStr
    # Clave de otra cuenta, opcional: se usa con el mismo modelo si la principal se queda sin cupo.
    llm_backup_api_key: SecretStr | None = None
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    # Modelos de respaldo si el principal está caído o no empieza a responder a tiempo. Vacío por defecto: en la evaluación
    # de la tesis todas las respuestas deben salir del mismo modelo.
    # Deben ser modelos :free: si el respaldo es de pago, OpenRouter lo cobra.
    llm_fallback_models: list[str] = []
    llm_reasoning_effort: str | None = "medium"
    llm_max_output_tokens: int = 8192
    llm_start_timeout_seconds: float = 150  # si el modelo no empieza a responder, se usa el respaldo

    max_history_messages: int = 6
    data_dir: Path = ROOT / "data"
    prompts_dir: Path = ROOT / "prompts"

    cors_origins: list[str] = ["http://localhost:3000"]

    # Servicio de recuperación (rag-istae-query) para B y C.
    retrieval_url: str = "http://127.0.0.1:8100"
    rag_api_key: SecretStr  # mismo nombre y valor que RAG_API_KEY del servicio de recuperación
    retrieval_timeout_seconds: float = 90  # C tarda hasta ~12 s en CPU; más si el servicio estaba dormido
    # Se envía siempre: la evaluación con expertos (Gold Standard v4) se hizo con reranker.
    retrieval_rerank: bool = True
    # A/AJ (contexto largo) quedan como históricos: no se cargan salvo que se pida.
    enable_long_context: bool = False
