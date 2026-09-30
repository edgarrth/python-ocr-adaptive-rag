from __future__ import annotations

from pathlib import Path
from typing import Any

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import ContextItem


class LightRagAdapter:
    """Integra LightRAG como motor graph-RAG alternativo para comparación."""

    def __init__(self, settings: Settings, canonical_dir: Path) -> None:
        self.settings = settings
        self.canonical_dir = canonical_dir
        self._rag: Any = None

    @property
    def available(self) -> bool:
        return self.settings.lightrag_enabled and bool(self.settings.openai_api_key)

    async def index(self, path: Path | None = None) -> bool:
        if not self.available:
            return False
        try:
            rag = await self._get_rag()
            paths = [path] if path else sorted(self.canonical_dir.glob("*.md"))
            for item in paths:
                await rag.ainsert(item.read_text(encoding="utf-8"), file_paths=str(item))
            return True
        except Exception:
            return False

    async def retrieve(self, question: str, top_k: int) -> list[ContextItem]:
        if not self.available:
            raise RuntimeError("LightRAG requiere LIGHTRAG_ENABLED=true y OPENAI_API_KEY")
        from lightrag import QueryParam

        rag = await self._get_rag()
        result = await rag.aquery(question, param=QueryParam(mode="hybrid", only_need_context=True, top_k=top_k))
        content = getattr(result, "content", None)
        text = content if content is not None else (result if isinstance(result, str) else str(result))
        return [
            ContextItem(
                id="lightrag-context",
                document_id="",
                source="LightRAG",
                title="Contexto híbrido de LightRAG",
                text=text,
                score=1.0,
                strategy="lightrag",
                entities=[],
                metadata={"mode": "hybrid"},
            )
        ]

    async def close(self) -> None:
        if self._rag is not None:
            await self._rag.finalize_storages()
            self._rag = None

    async def _get_rag(self) -> Any:
        if self._rag is not None:
            return self._rag
        from lightrag import LightRAG
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc

        self.settings.lightrag_workdir.mkdir(parents=True, exist_ok=True)

        async def llm_func(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **kwargs: Any) -> str:
            return await openai_complete_if_cache(
                self.settings.openai_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                **kwargs,
            )

        async def embedding_func(texts: list[str]) -> Any:
            return await openai_embed(
                texts,
                model=self.settings.openai_embedding_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )

        rag = LightRAG(
            working_dir=str(self.settings.lightrag_workdir),
            llm_model_func=llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.settings.openai_embedding_dimension,
                max_token_size=8191,
                func=embedding_func,
            ),
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()
        self._rag = rag
        return rag
