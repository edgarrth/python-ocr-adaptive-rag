from __future__ import annotations

from qdrant_client import QdrantClient, models

from pe.axiz.payment_knowledge.domain.models import ContextItem, DocumentChunk
from pe.axiz.payment_knowledge.infrastructure.embeddings import FastEmbedEmbedder


class QdrantStore:
    """Persistencia vectorial común para RAG nativo e híbrido."""

    def __init__(self, url: str, collection: str, embedder: FastEmbedEmbedder, dimension: int) -> None:
        self.client = QdrantClient(url=url, timeout=20)
        self.collection = collection
        self.embedder = embedder
        self.dimension = dimension

    def close(self) -> None:
        self.client.close()

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def ensure_collection(self) -> None:
        collections = {item.name for item in self.client.get_collections().collections}
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
            )

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self.ensure_collection()
        vectors = self.embedder.embed(chunk.text for chunk in chunks)
        points = [
            models.PointStruct(id=chunk.id, vector=vector, payload=chunk.model_dump(mode="json"))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, query: str, limit: int) -> list[ContextItem]:
        self.ensure_collection()
        vector = self.embedder.embed([query])[0]
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        contexts: list[ContextItem] = []
        for point in response.points:
            payload = point.payload or {}
            contexts.append(
                ContextItem(
                    id=str(payload.get("id", point.id)),
                    document_id=str(payload.get("document_id", "")),
                    source=str(payload.get("source", "")),
                    title=str(payload.get("title", "")),
                    text=str(payload.get("text", "")),
                    score=float(point.score or 0.0),
                    strategy="vector",
                    entities=list(payload.get("entities") or []),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return contexts

    def all_contexts(self, limit: int = 500) -> list[ContextItem]:
        self.ensure_collection()
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        result: list[ContextItem] = []
        for point in points:
            payload = point.payload or {}
            result.append(
                ContextItem(
                    id=str(payload.get("id", point.id)),
                    document_id=str(payload.get("document_id", "")),
                    source=str(payload.get("source", "")),
                    title=str(payload.get("title", "")),
                    text=str(payload.get("text", "")),
                    score=0.0,
                    strategy="corpus",
                    entities=list(payload.get("entities") or []),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return result
