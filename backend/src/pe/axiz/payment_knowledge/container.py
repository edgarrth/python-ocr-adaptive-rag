from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pe.axiz.payment_knowledge.application.services import EvaluationService, IngestionService, QueryService
from pe.axiz.payment_knowledge.config import get_settings
from pe.axiz.payment_knowledge.generation.llm import AnswerGenerator
from pe.axiz.payment_knowledge.infrastructure.document_processing import DocumentProcessor, PaymentEntityExtractor, SemanticChunker
from pe.axiz.payment_knowledge.infrastructure.embeddings import FastEmbedEmbedder
from pe.axiz.payment_knowledge.infrastructure.memgraph_store import MemgraphStore
from pe.axiz.payment_knowledge.infrastructure.qdrant_store import QdrantStore
from pe.axiz.payment_knowledge.retrieval.graphrag import GraphRagRetriever
from pe.axiz.payment_knowledge.retrieval.lightrag_adapter import LightRagAdapter
from pe.axiz.payment_knowledge.retrieval.native import HybridRagRetriever, NativeRagRetriever
from pe.axiz.payment_knowledge.retrieval.raglight_adapter import RagLightAdapter
from pe.axiz.payment_knowledge.retrieval.router import AdaptiveRouter


class AppContainer:
    def __init__(self) -> None:
        settings = get_settings()
        embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_dimension)
        qdrant = QdrantStore(settings.qdrant_url, settings.qdrant_collection, embedder, settings.embedding_dimension)
        memgraph = MemgraphStore(settings.memgraph_uri, settings.memgraph_user, settings.memgraph_password)
        canonical = Path(".runtime/canonical")
        raglight = RagLightAdapter(settings, canonical)
        lightrag = LightRagAdapter(settings, canonical)
        native = NativeRagRetriever(qdrant)
        hybrid = HybridRagRetriever(qdrant)
        graphrag = GraphRagRetriever(memgraph, hybrid)
        query = QueryService(native, hybrid, graphrag, raglight, lightrag, AdaptiveRouter(), AnswerGenerator(settings))
        self.settings = settings
        self.qdrant = qdrant
        self.memgraph = memgraph
        self.raglight = raglight
        self.lightrag = lightrag
        self.ingestion = IngestionService(
            settings,
            DocumentProcessor(settings),
            SemanticChunker(PaymentEntityExtractor()),
            qdrant,
            memgraph,
            raglight,
            lightrag,
            canonical,
        )
        self.query = query
        self.evaluation = EvaluationService(query, Path("datasets/evaluation/questions.json"))

    async def close(self) -> None:
        await self.lightrag.close()
        self.memgraph.close()
        self.qdrant.close()


@lru_cache(maxsize=1)
def get_container() -> AppContainer:
    return AppContainer()
