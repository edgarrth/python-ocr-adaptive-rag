from functools import lru_cache
from pathlib import Path
import json

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración central de la PoC."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "Axiz Adaptive RAG Payments"
    app_env: str = "local"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:4200"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "axiz_payment_chunks"
    raglight_collection: str = "axiz_payment_raglight"
    embedding_provider: str = "fastembed"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    hf_token: str = ""

    memgraph_uri: str = "bolt://localhost:7687"
    memgraph_user: str = ""
    memgraph_password: str = ""

    llm_provider: str = "extractive"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimension: int = 1536

    glm_ocr_mode: str = "selfhosted"
    glm_ocr_api_url: str = "http://localhost:8080/v1/chat/completions"
    glm_ocr_model: str = "glm-ocr"
    glm_ocr_layout_device: str = "cpu"
    glm_ocr_max_workers: int = 1
    glm_ocr_request_timeout_seconds: int = 600
    glm_ocr_max_tokens: int = 2048
    glm_ocr_image_max_side: int = 1400
    glm_ocr_image_max_pixels: int = 1600000
    glm_ocr_image_quality: int = 92
    glm_ocr_pdf_scale: float = 1.5
    zhipu_api_key: str = ""
    ocr_policy: str = "auto"
    ocr_min_text_chars: int = 120

    raglight_enabled: bool = True
    raglight_service_url: str = "http://localhost:8010"
    raglight_timeout_seconds: float = 600.0
    lightrag_enabled: bool = True
    lightrag_workdir: Path = Path(".runtime/lightrag")
    lightrag_timeout_seconds: float = 300.0
    datasets_dir: Path = Path("datasets/sample_documents")

    retrieval_candidate_multiplier: int = 4
    retrieval_candidate_minimum: int = 12
    semantic_dedup_threshold: float = 0.94
    rerank_semantic_weight: float = 0.65
    rerank_lexical_weight: float = 0.20
    rerank_original_weight: float = 0.10
    rerank_entity_weight: float = 0.05

    ingestion_job_workers: int = 1
    ingestion_job_queue_size: int = 32
    jobs_dir: Path = Path(".runtime/jobs")

    def parsed_cors_origins(self) -> list[str]:
        """Acepta CORS_ORIGINS como CSV o como arreglo JSON sin depender del decoder de settings."""
        raw = self.cors_origins.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = None
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
        return [part.strip() for part in raw.split(",") if part.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
