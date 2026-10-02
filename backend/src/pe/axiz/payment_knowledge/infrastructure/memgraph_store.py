from __future__ import annotations

import re
import unicodedata

from neo4j import GraphDatabase

from pe.axiz.payment_knowledge.domain.models import ContextItem, DocumentChunk


class MemgraphStore:
    """Acceso a Memgraph mediante Bolt y expansión de conocimiento por relaciones."""

    def __init__(self, uri: str, user: str = "", password: str = "") -> None:
        auth = (user, password) if user else None
        self.driver = GraphDatabase.driver(uri, auth=auth, connection_timeout=5.0)

    def close(self) -> None:
        self.driver.close()

    def ping(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def upsert_document(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        document_id = chunks[0].document_id
        source = chunks[0].source
        title = chunks[0].title
        with self.driver.session() as session:
            session.run(
                "MERGE (d:Document {id:$id}) SET d.source=$source, d.title=$title",
                id=document_id,
                source=source,
                title=title,
            ).consume()
            for chunk in chunks:
                session.run(
                    """
                    MATCH (d:Document {id:$document_id})
                    MERGE (c:Chunk {id:$chunk_id})
                    SET c.document_id=$document_id, c.text=$text, c.source=$source,
                        c.title=$title, c.position=$position
                    MERGE (d)-[:CONTAINS]->(c)
                    WITH c
                    UNWIND $entities AS entity_name
                    MERGE (e:Entity {name:entity_name})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    text=chunk.text,
                    source=chunk.source,
                    title=chunk.title,
                    position=chunk.position,
                    entities=chunk.entities,
                ).consume()
                if len(chunk.entities) > 1:
                    session.run(
                        """
                        MATCH (c:Chunk {id:$chunk_id})-[:MENTIONS]->(a:Entity)
                        MATCH (c)-[:MENTIONS]->(b:Entity)
                        WHERE a.name < b.name
                        MERGE (a)-[:CO_OCCURS {chunk_id:$chunk_id}]->(b)
                        """,
                        chunk_id=chunk.id,
                    ).consume()

    def graph_search(self, question: str, limit: int) -> list[ContextItem]:
        """Busca entidades semilla y expande un salto para recuperar evidencia conectada."""
        terms = self._terms(question)
        with self.driver.session() as session:
            records = session.run(
                """
                MATCH (seed:Entity)
                WHERE any(term IN $terms WHERE toLower(seed.name) CONTAINS term)
                MATCH (direct:Chunk)-[:MENTIONS]->(seed)
                OPTIONAL MATCH (seed)-[rel:CO_OCCURS]-(neighbor:Entity)<-[:MENTIONS]-(expanded:Chunk)
                WITH collect(DISTINCT {
                    id: direct.id,
                    document_id: direct.document_id,
                    source: direct.source,
                    title: direct.title,
                    text: direct.text,
                    position: direct.position,
                    score: 3.0
                }) + collect(DISTINCT CASE WHEN expanded IS NULL THEN null ELSE {
                    id: expanded.id,
                    document_id: expanded.document_id,
                    source: expanded.source,
                    title: expanded.title,
                    text: expanded.text,
                    position: expanded.position,
                    score: 1.5
                } END) AS rows
                UNWIND rows AS row
                WITH row WHERE row IS NOT NULL
                RETURN row.id AS id, row.document_id AS document_id, row.source AS source,
                       row.title AS title, row.text AS text, row.position AS position,
                       max(row.score) AS score
                ORDER BY score DESC
                LIMIT $limit
                """,
                terms=terms,
                limit=limit,
            )
            result = [self._context(record) for record in records]

            if result:
                return result

            fallback = session.run(
                """
                MATCH (c:Chunk)
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                WITH c, collect(DISTINCT e.name) AS entities
                WHERE any(term IN $terms WHERE toLower(c.text) CONTAINS term)
                   OR any(entity IN entities WHERE any(term IN $terms WHERE toLower(entity) CONTAINS term))
                RETURN c.id AS id, c.document_id AS document_id, c.source AS source,
                       c.title AS title, c.text AS text, c.position AS position,
                       1.0 AS score
                LIMIT $limit
                """,
                terms=terms,
                limit=limit,
            )
            return [self._context(record) for record in fallback]

    def neighborhood(self, entity: str, limit: int = 20) -> dict[str, list[dict[str, object]]]:
        """Expande vecinos con matching tolerante a mayúsculas y acentos."""
        normalized_query = self._normalize_search_text(entity)
        with self.driver.session() as session:
            candidates = session.run("MATCH (e:Entity) RETURN e.name AS name")
            matched_names = [
                str(record["name"])
                for record in candidates
                if normalized_query in self._normalize_search_text(str(record["name"]))
            ]
            if not matched_names:
                return {"edges": []}

            records = session.run(
                """
                MATCH (a:Entity)-[r:CO_OCCURS]-(b:Entity)
                WHERE a.name IN $entities
                RETURN a.name AS source, b.name AS target, count(r) AS weight
                ORDER BY weight DESC LIMIT $limit
                """,
                entities=matched_names,
                limit=limit,
            )
            return {"edges": [dict(record) for record in records]}

    @staticmethod
    def _normalize_search_text(value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", value)
        return "".join(
            char for char in decomposed.lower() if not unicodedata.combining(char)
        ).strip()

    @staticmethod
    def _context(record: object) -> ContextItem:
        return ContextItem(
            id=record["id"],
            document_id=record["document_id"] or "",
            source=record["source"],
            title=record["title"],
            text=record["text"],
            score=float(record["score"] or 0.0),
            strategy="graph",
            entities=[],
            metadata={"position": record["position"], "expansion_hops": 1},
        )

    @staticmethod
    def _terms(question: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9áéíóúñ_-]{3,}", question.lower())
        stop = {"que", "como", "para", "por", "con", "del", "las", "los", "una", "sobre", "esta", "este"}
        return [token for token in tokens if token not in stop][:12] or [question.lower()]
