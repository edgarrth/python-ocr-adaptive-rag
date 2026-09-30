from __future__ import annotations

from pathlib import Path

import httpx

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import ContextItem


class RagLightAdapter:
    """Consume el servicio aislado de RAGLight para evitar conflictos de dependencias."""

    def __init__(self, settings: Settings, canonical_dir: Path) -> None:
        self.settings = settings
        self.canonical_dir = canonical_dir
        self._client = httpx.Client(
            base_url=self.settings.raglight_service_url.rstrip("/"),
            timeout=self.settings.raglight_timeout_seconds,
        )

    @property
    def available(self) -> bool:
        return self.settings.raglight_enabled

    def ping(self) -> bool:
        if not self.available:
            return False
        try:
            response = self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def index(self, path: Path | None = None) -> bool:
        if not self.available:
            return False
        target = (path or self.canonical_dir).resolve()
        try:
            response = self._client.post("/index", json={"path": str(target)})
            response.raise_for_status()
            return bool(response.json().get("indexed", False))
        except (httpx.HTTPError, ValueError):
            return False

    def retrieve(self, question: str, top_k: int) -> list[ContextItem]:
        if not self.available:
            raise RuntimeError("RAGLight está deshabilitado")

        response = self._client.post(
            "/search",
            json={"question": question, "top_k": top_k},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("RAGLight devolvió un payload de búsqueda inesperado")

        result: list[ContextItem] = []
        for index, item in enumerate(payload):
            metadata = item.get("metadata", {}) or {}
            source = self._source_name(str(item.get("source", "raglight")))
            result.append(
                ContextItem(
                    id=str(item.get("id", f"raglight-{index}")),
                    document_id=str(item.get("document_id", "")),
                    source=Path(source).name,
                    title=str(item.get("title", Path(source).stem)),
                    text=str(item.get("text", "")),
                    score=float(item.get("score", 1.0 / (index + 1))),
                    strategy="raglight",
                    entities=[],
                    metadata=dict(metadata),
                )
            )
        return result

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _source_name(source: str) -> str:
        name = Path(source).name
        if "__" in name:
            name = name.split("__", 1)[1]
        if name.endswith(".md") and name.count(".") > 1:
            name = name[:-3]
        return name
