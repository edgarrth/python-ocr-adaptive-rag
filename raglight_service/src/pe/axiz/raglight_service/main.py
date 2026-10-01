from __future__ import annotations

import logging
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logger = logging.getLogger("pe.axiz.raglight_service")


def configure_huggingface_environment() -> bool:
    """Propaga el token a nombres reconocidos por distintas capas de Hugging Face."""
    token = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
    )
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    return True


HF_TOKEN_CONFIGURED = configure_huggingface_environment()


def dependency_versions() -> dict[str, str]:
    """Expone versiones útiles para diagnosticar compatibilidad sin revelar secretos."""
    packages = (
        "raglight",
        "langgraph",
        "langgraph-prebuilt",
        "langgraph-checkpoint",
        "langgraph-sdk",
        "qdrant-client",
    )
    resolved: dict[str, str] = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


class HealthResponse(BaseModel):
    status: str
    hf_token_configured: bool
    vector_store_ready: bool
    dependencies: dict[str, str]


class ReadyResponse(BaseModel):
    ready: bool
    hf_token_configured: bool
    collection: str
    points: int


class IndexRequest(BaseModel):
    path: str


class IndexResponse(BaseModel):
    indexed: bool
    files: int
    points_before: int
    points_after: int
    detail: str


class ResetResponse(BaseModel):
    reset: bool
    deleted_collections: list[str]


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchItem(BaseModel):
    id: str
    document_id: str = ""
    source: str
    title: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagLightEngine:
    """Encapsula RAGLight para aislar su árbol de dependencias del backend principal."""

    SUPPORTED_SUFFIXES = {".md", ".txt", ".html", ".pdf"}

    def __init__(self) -> None:
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = os.getenv("RAGLIGHT_COLLECTION", "axiz_payment_raglight")
        self.embedding_model = os.getenv(
            "RAGLIGHT_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.canonical_dir = Path(
            os.getenv("RAGLIGHT_CANONICAL_DIR", ".runtime/canonical")
        ).resolve()
        self._vector_store: Any = None
        self._lock = Lock()

    def ready(self) -> ReadyResponse:
        """Inicializa embeddings + Qdrant para fallar temprano antes de procesar el dataset."""
        store = self._get_vector_store()
        return ReadyResponse(
            ready=True,
            hf_token_configured=HF_TOKEN_CONFIGURED,
            collection=self.collection,
            points=self._count_points(store),
        )

    def index(self, requested_path: str) -> IndexResponse:
        path = Path(requested_path).resolve()
        files = self._validate_directory(path)
        store = self._get_vector_store()

        with self._lock:
            points_before = self._count_points(store)
            store.ingest(data_path=str(path))
            points_after = self._count_points(store)

        added = points_after - points_before
        if added < len(files):
            raise RuntimeError(
                "RAGLight no generó suficientes puntos para los documentos enviados: "
                f"archivos={len(files)}, puntos_agregados={added}"
            )

        return IndexResponse(
            indexed=True,
            files=len(files),
            points_before=points_before,
            points_after=points_after,
            detail=f"RAGLight indexó {len(files)} archivo(s) y agregó {added} punto(s)",
        )

    def reset(self) -> ResetResponse:
        """Elimina solo las colecciones administradas por RAGLight y reinicia su estado local."""
        from qdrant_client import QdrantClient

        parsed = urlparse(self.qdrant_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 6333)
        deleted: list[str] = []

        with self._lock:
            if self._vector_store is not None:
                try:
                    self._vector_store.client.close()
                finally:
                    self._vector_store = None

            client = QdrantClient(host=host, port=port)
            try:
                existing = {item.name for item in client.get_collections().collections}
                for name in (self.collection, f"{self.collection}_classes"):
                    if name in existing:
                        client.delete_collection(collection_name=name)
                        deleted.append(name)
            finally:
                client.close()

        return ResetResponse(reset=True, deleted_collections=deleted)

    def search(self, question: str, top_k: int) -> list[SearchItem]:
        store = self._get_vector_store()
        with self._lock:
            docs = store.similarity_search(question, k=top_k)

        result: list[SearchItem] = []
        for index, doc in enumerate(docs):
            text = getattr(doc, "page_content", None) or getattr(doc, "text", None) or str(doc)
            metadata = getattr(doc, "metadata", {}) or {}
            source = self._source_name(str(metadata.get("source", "raglight")))
            result.append(
                SearchItem(
                    id=str(metadata.get("id", f"raglight-{index}")),
                    document_id=str(metadata.get("document_id", "")),
                    source=Path(source).name,
                    title=str(metadata.get("title", Path(source).stem)),
                    text=str(text),
                    score=float(metadata.get("score", 1.0 / (index + 1))),
                    metadata=dict(metadata),
                )
            )
        return result

    def _validate_directory(self, path: Path) -> list[Path]:
        if not path.exists() or not path.is_dir():
            raise ValueError(f"El path de RAGLight debe ser un directorio existente: {path}")
        try:
            path.relative_to(self.canonical_dir)
        except ValueError as exc:
            raise ValueError(
                f"El directorio debe estar dentro de {self.canonical_dir}"
            ) from exc

        files = [
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_SUFFIXES
        ]
        if not files:
            raise ValueError(f"No hay documentos soportados para indexar en {path}")
        return files

    def _get_vector_store(self) -> Any:
        if self._vector_store is not None:
            return self._vector_store

        from raglight.config.settings import Settings
        from raglight.rag.builder import Builder

        parsed = urlparse(self.qdrant_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 6333)

        self._vector_store = (
            Builder()
            .with_embeddings(Settings.HUGGINGFACE, model_name=self.embedding_model)
            .with_vector_store(
                Settings.QDRANT,
                host=host,
                port=port,
                collection_name=self.collection,
                search_type=Settings.SEARCH_HYBRID,
                alpha=0.5,
            )
            .build_vector_store()
        )
        return self._vector_store

    def _count_points(self, store: Any) -> int:
        result = store.client.count(collection_name=self.collection, exact=True)
        return int(result.count)

    @staticmethod
    def _source_name(source: str) -> str:
        name = Path(source).name
        if "__" in name:
            name = name.split("__", 1)[1]
        return name


engine = RagLightEngine()
app = FastAPI(title="Axiz RAGLight Retrieval Service", version="0.2.3")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        hf_token_configured=HF_TOKEN_CONFIGURED,
        vector_store_ready=engine._vector_store is not None,
        dependencies=dependency_versions(),
    )


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    try:
        return engine.ready()
    except Exception as exc:
        logger.exception("Falló la preparación de RAGLight")
        raise HTTPException(
            status_code=503,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.post("/reset", response_model=ResetResponse)
def reset_index() -> ResetResponse:
    try:
        return engine.reset()
    except Exception as exc:
        logger.exception("Falló el reset de RAGLight")
        raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/index", response_model=IndexResponse)
def index_documents(request: IndexRequest) -> IndexResponse:
    try:
        return engine.index(request.path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Falló la indexación de RAGLight")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/search", response_model=list[SearchItem])
def search(request: SearchRequest) -> list[SearchItem]:
    try:
        return engine.search(request.question, request.top_k)
    except Exception as exc:
        logger.exception("Falló la búsqueda de RAGLight")
        raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}") from exc
