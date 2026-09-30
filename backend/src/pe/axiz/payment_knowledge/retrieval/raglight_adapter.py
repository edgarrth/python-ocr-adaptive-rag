from __future__ import annotations

from pathlib import Path
from typing import Any

from pe.axiz.payment_knowledge.config import Settings as AppSettings
from pe.axiz.payment_knowledge.domain.models import ContextItem


class RagLightAdapter:
    """Integra RAGLight como flujo alterno sobre una colección Qdrant dedicada."""

    def __init__(self, settings: AppSettings, canonical_dir: Path) -> None:
        self.settings = settings
        self.canonical_dir = canonical_dir
        self._vector_store: Any = None

    @property
    def available(self) -> bool:
        return self.settings.raglight_enabled

    def index(self, path: Path | None = None) -> bool:
        if not self.available:
            return False
        try:
            store = self._get_vector_store()
            store.ingest(data_path=str(path or self.canonical_dir))
            return True
        except Exception:
            return False

    def retrieve(self, question: str, top_k: int) -> list[ContextItem]:
        if not self.available:
            raise RuntimeError("RAGLight está deshabilitado")
        store = self._get_vector_store()
        docs = store.similarity_search(question, k=top_k)
        result: list[ContextItem] = []
        for index, doc in enumerate(docs):
            text = getattr(doc, "page_content", None) or getattr(doc, "text", None) or str(doc)
            metadata = getattr(doc, "metadata", {}) or {}
            source = self._source_name(str(metadata.get("source", "raglight")))
            result.append(
                ContextItem(
                    id=str(metadata.get("id", f"raglight-{index}")),
                    document_id=str(metadata.get("document_id", "")),
                    source=Path(source).name,
                    title=str(metadata.get("title", Path(source).stem)),
                    text=str(text),
                    score=float(metadata.get("score", 1.0 / (index + 1))),
                    strategy="raglight",
                    entities=[],
                    metadata=dict(metadata),
                )
            )
        return result


    @staticmethod
    def _source_name(source: str) -> str:
        name = Path(source).name
        if "__" in name:
            name = name.split("__", 1)[1]
        if name.endswith(".md") and name.count(".") > 1:
            name = name[:-3]
        return name

    def _get_vector_store(self) -> Any:
        if self._vector_store is not None:
            return self._vector_store
        from raglight.config.settings import Settings
        from raglight.rag.builder import Builder

        qdrant_host = self.settings.qdrant_url.replace("http://", "").replace("https://", "").split(":")[0]
        qdrant_port = int(self.settings.qdrant_url.rsplit(":", 1)[-1])
        self._vector_store = (
            Builder()
            .with_embeddings(Settings.HUGGINGFACE, model_name=self.settings.embedding_model)
            .with_vector_store(
                Settings.QDRANT,
                host=qdrant_host,
                port=qdrant_port,
                collection_name=self.settings.raglight_collection,
                search_type=Settings.SEARCH_HYBRID,
                alpha=0.5,
            )
            .build_vector_store()
        )
        return self._vector_store
