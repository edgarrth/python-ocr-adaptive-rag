from __future__ import annotations

from pathlib import Path

import httpx

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import ContextItem
from pe.axiz.payment_knowledge.retrieval.source_attribution import (
    canonical_source_name,
    infer_source_from_text,
    is_generic_source,
)


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

    def ready(self) -> bool:
        if not self.available:
            return False
        response = self._client.get("/ready")
        self._raise_for_status(response, "ready")
        payload = response.json()
        if payload.get("ready") is not True:
            raise RuntimeError(f"RAGLight no confirmó readiness: {payload}")
        return True

    def reset(self) -> None:
        if not self.available:
            return
        response = self._client.post("/reset")
        self._raise_for_status(response, "reset")
        payload = response.json()
        if payload.get("reset") is not True:
            raise RuntimeError("RAGLight no confirmó el reinicio de sus colecciones")

    def index(self, directory: Path | None = None) -> bool:
        if not self.available:
            return False
        target = (directory or self.canonical_dir).resolve()
        response = self._client.post("/index", json={"path": str(target)})
        self._raise_for_status(response, "index")
        payload = response.json()
        if payload.get("indexed") is not True:
            raise RuntimeError(f"RAGLight no confirmó la indexación: {payload}")
        return True

    def retrieve(self, question: str, top_k: int) -> list[ContextItem]:
        if not self.available:
            raise RuntimeError("RAGLight está deshabilitado")

        response = self._client.post(
            "/search",
            json={"question": question, "top_k": top_k},
        )
        self._raise_for_status(response, "search")
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("RAGLight devolvió un payload de búsqueda inesperado")

        result: list[ContextItem] = []
        for index, item in enumerate(payload):
            metadata = dict(item.get("metadata", {}) or {})
            text = str(item.get("text", ""))
            raw_source = str(item.get("source", "raglight"))
            source = canonical_source_name(raw_source)
            document_id = str(item.get("document_id", ""))

            if is_generic_source(source):
                inferred = infer_source_from_text(text, self.canonical_dir)
                if inferred is not None:
                    source = inferred.source
                    document_id = document_id or inferred.document_id
                    metadata.update(
                        {
                            "canonical_name": inferred.canonical_name,
                            "source_attribution": inferred.method,
                            "source_attribution_score": inferred.score,
                        }
                    )
                else:
                    metadata.setdefault("source_attribution", "unavailable")

            raw_title = str(item.get("title", "")).strip()
            title = (
                Path(source).stem
                if not raw_title or raw_title.lower() in {"raglight", "document", "unknown"}
                else raw_title
            )
            result.append(
                ContextItem(
                    id=str(item.get("id", f"raglight-{index}")),
                    document_id=document_id,
                    source=Path(source).name,
                    title=title,
                    text=text,
                    score=float(item.get("score", 1.0 / (index + 1))),
                    strategy="raglight",
                    entities=[],
                    metadata=metadata,
                )
            )
        return result


    @staticmethod
    def _raise_for_status(response: httpx.Response, operation: str) -> None:
        if not response.is_error:
            return
        detail = response.text
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("detail"):
                detail = str(payload["detail"])
        except ValueError:
            pass
        raise RuntimeError(
            f"RAGLight {operation} falló con HTTP {response.status_code}: {detail}"
        )

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _source_name(source: str) -> str:
        return canonical_source_name(source)
