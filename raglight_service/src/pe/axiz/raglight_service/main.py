from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    path: str


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

    def index(self, requested_path: str) -> bool:
        path = Path(requested_path).resolve()
        self._validate_path(path)
        store = self._get_vector_store()
        with self._lock:
            store.ingest(data_path=str(path))
        return True

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

    def _validate_path(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise ValueError(f"El documento canónico no existe: {path}")
        try:
            path.relative_to(self.canonical_dir)
        except ValueError as exc:
            raise ValueError(
                f"El documento debe estar dentro de {self.canonical_dir}"
            ) from exc

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

    @staticmethod
    def _source_name(source: str) -> str:
        name = Path(source).name
        if "__" in name:
            name = name.split("__", 1)[1]
        if name.endswith(".md") and name.count(".") > 1:
            name = name[:-3]
        return name


engine = RagLightEngine()
app = FastAPI(title="Axiz RAGLight Retrieval Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index")
def index_document(request: IndexRequest) -> dict[str, bool]:
    try:
        return {"indexed": engine.index(request.path)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/search", response_model=list[SearchItem])
def search(request: SearchRequest) -> list[SearchItem]:
    try:
        return engine.search(request.question, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
