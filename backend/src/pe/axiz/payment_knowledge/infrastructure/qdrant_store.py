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

    @staticmethod
    def _document_filter(document_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

    def count_document(self, document_id: str) -> int:
        self.ensure_collection()
        result = self.client.count(
            collection_name=self.collection,
            count_filter=self._document_filter(document_id),
            exact=True,
        )
        return int(result.count)

    def document_exists(self, document_id: str) -> bool:
        return self.count_document(document_id) > 0

    def delete_document(self, document_id: str) -> None:
        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self._document_filter(document_id)),
            wait=True,
        )

    def replace_document(self, chunks: list[DocumentChunk]) -> None:
        """Reemplaza el documento completo para evitar chunks huérfanos o duplicados."""
        if not chunks:
            return
        document_id = chunks[0].document_id
        self.delete_document(document_id)
        self.upsert(chunks)

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
