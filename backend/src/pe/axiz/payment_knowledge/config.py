from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración central de la PoC."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "Axiz Adaptive RAG Payments"
    app_env: str = "local"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:4200"]

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "axiz_payment_chunks"
    raglight_collection: str = "axiz_payment_raglight"
    embedding_provider: str = "fastembed"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384

    memgraph_uri: str = "bolt://localhost:7687"
    memgraph_user: str = ""
    memgraph_password: str = ""

    llm_provider: str = "extractive"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimension: int = 1536

    glm_ocr_mode: str = "maas"
    glm_ocr_api_url: str = ""
    zhipu_api_key: str = ""
    ocr_min_text_chars: int = 120

    raglight_enabled: bool = True
    lightrag_enabled: bool = True
    lightrag_workdir: Path = Path(".runtime/lightrag")
    datasets_dir: Path = Path("datasets/sample_documents")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
