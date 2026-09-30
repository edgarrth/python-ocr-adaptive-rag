from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import (
    ContextItem,
    EvaluationItem,
    EvaluationRequest,
    EvaluationResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievalStrategy,
)
from pe.axiz.payment_knowledge.generation.llm import AnswerGenerator
from pe.axiz.payment_knowledge.infrastructure.document_processing import DocumentProcessor, PaymentEntityExtractor, SemanticChunker
from pe.axiz.payment_knowledge.infrastructure.memgraph_store import MemgraphStore
from pe.axiz.payment_knowledge.infrastructure.qdrant_store import QdrantStore
from pe.axiz.payment_knowledge.retrieval.graphrag import GraphRagRetriever
from pe.axiz.payment_knowledge.retrieval.lightrag_adapter import LightRagAdapter
from pe.axiz.payment_knowledge.retrieval.native import HybridRagRetriever, NativeRagRetriever
from pe.axiz.payment_knowledge.retrieval.raglight_adapter import RagLightAdapter
from pe.axiz.payment_knowledge.retrieval.router import AdaptiveRouter


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        processor: DocumentProcessor,
        chunker: SemanticChunker,
        qdrant: QdrantStore,
        memgraph: MemgraphStore,
        raglight: RagLightAdapter,
        lightrag: LightRagAdapter,
        canonical_dir: Path,
    ) -> None:
        self.settings = settings
        self.processor = processor
        self.chunker = chunker
        self.qdrant = qdrant
        self.memgraph = memgraph
        self.raglight = raglight
        self.lightrag = lightrag
        self.canonical_dir = canonical_dir
        self.canonical_dir.mkdir(parents=True, exist_ok=True)

    async def ingest_path(self, path: Path) -> IngestResponse:
        document = await asyncio.to_thread(self.processor.parse, path)
        chunks = self.chunker.split(document)
        await asyncio.to_thread(self.qdrant.upsert, chunks)
        await asyncio.to_thread(self.memgraph.upsert_document, chunks)

        canonical_path = self.canonical_dir / f"{document.document_id}__{document.source_name}.md"
        canonical_path.write_text(document.markdown, encoding="utf-8")
        raglight_indexed = await asyncio.to_thread(self.raglight.index, canonical_path)
        lightrag_indexed = await self.lightrag.index(canonical_path)
        return IngestResponse(
            document_id=document.document_id,
            source=document.source_name,
            processor=document.processor,
            chunks=len(chunks),
            entities=len({entity for chunk in chunks for entity in chunk.entities}),
            ocr_used=document.ocr_used,
            raglight_indexed=raglight_indexed,
            lightrag_indexed=lightrag_indexed,
        )

    async def ingest_dataset(self) -> list[IngestResponse]:
        results = []
        for path in sorted(self.settings.datasets_dir.glob("*")):
            if path.is_file():
                results.append(await self.ingest_path(path))
        return results


class QueryService:
    def __init__(
        self,
        native: NativeRagRetriever,
        hybrid: HybridRagRetriever,
        graphrag: GraphRagRetriever,
        raglight: RagLightAdapter,
        lightrag: LightRagAdapter,
        router: AdaptiveRouter,
        generator: AnswerGenerator,
    ) -> None:
        self.native = native
        self.hybrid = hybrid
        self.graphrag = graphrag
        self.raglight = raglight
        self.lightrag = lightrag
        self.router = router
        self.generator = generator

    async def query(self, request: QueryRequest) -> QueryResponse:
        total_start = time.perf_counter()
        route_trace: dict[str, object] = {}
        executed = request.strategy
        if request.strategy == RetrievalStrategy.AUTO:
            executed, route_trace = self.router.route(request.question)

        retrieval_start = time.perf_counter()
        contexts = await self._retrieve(executed, request.question, request.top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        generation_start = time.perf_counter()
        answer = await asyncio.to_thread(self.generator.generate, request.question, contexts)
        generation_ms = (time.perf_counter() - generation_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000

        trace = {
            "router": route_trace,
            "requested_strategy": request.strategy,
            "executed_strategy": executed,
            "returned_contexts": len(contexts),
            "sources": sorted({item.source for item in contexts}),
        }
        return QueryResponse(
            answer=answer,
            requested_strategy=request.strategy,
            executed_strategy=executed,
            contexts=contexts if request.include_evidence else [],
            trace=trace if request.include_trace else {},
            timings_ms={
                "retrieval": round(retrieval_ms, 2),
                "generation": round(generation_ms, 2),
                "total": round(total_ms, 2),
            },
        )

    async def retrieve_contexts(self, strategy: RetrievalStrategy, question: str, top_k: int) -> list[ContextItem]:
        executed = strategy
        if strategy == RetrievalStrategy.AUTO:
            executed, _ = self.router.route(question)
        return await self._retrieve(executed, question, top_k)

    async def _retrieve(self, strategy: RetrievalStrategy, question: str, top_k: int) -> list[ContextItem]:
        if strategy == RetrievalStrategy.NATIVE_RAG:
            return await asyncio.to_thread(self.native.retrieve, question, top_k)
        if strategy == RetrievalStrategy.HYBRID_RAG:
            return await asyncio.to_thread(self.hybrid.retrieve, question, top_k)
        if strategy == RetrievalStrategy.GRAPHRAG:
            return await asyncio.to_thread(self.graphrag.retrieve, question, top_k)
        if strategy == RetrievalStrategy.RAGLIGHT:
            return await asyncio.to_thread(self.raglight.retrieve, question, top_k)
        if strategy == RetrievalStrategy.LIGHTRAG:
            return await self.lightrag.retrieve(question, top_k)
        raise ValueError(f"Estrategia no soportada: {strategy}")


class EvaluationService:
    def __init__(self, query_service: QueryService, dataset_file: Path) -> None:
        self.query_service = query_service
        self.dataset_file = dataset_file

    async def run(self, request: EvaluationRequest) -> EvaluationResponse:
        cases = json.loads(self.dataset_file.read_text(encoding="utf-8"))
        items: list[EvaluationItem] = []
        for case in cases:
            for strategy in request.strategies:
                started = time.perf_counter()
                expected = case["expected_sources"]
                try:
                    contexts = await self.query_service.retrieve_contexts(strategy, case["question"], request.top_k)
                    latency = (time.perf_counter() - started) * 1000
                    sources = [item.source for item in contexts]
                    rank = next((index for index, source in enumerate(sources, 1) if source in expected), None)
                    items.append(
                        EvaluationItem(
                            question=case["question"],
                            strategy=strategy,
                            expected_sources=expected,
                            retrieved_sources=sources,
                            hit=rank is not None,
                            reciprocal_rank=(1.0 / rank) if rank else 0.0,
                            latency_ms=round(latency, 2),
                        )
                    )
                except Exception as exc:
                    latency = (time.perf_counter() - started) * 1000
                    items.append(
                        EvaluationItem(
                            question=case["question"],
                            strategy=strategy,
                            expected_sources=expected,
                            retrieved_sources=[],
                            hit=False,
                            reciprocal_rank=0.0,
                            latency_ms=round(latency, 2),
                            success=False,
                            error=str(exc),
                        )
                    )
        summary: dict[str, dict[str, float]] = {}
        for strategy in request.strategies:
            selected = [item for item in items if item.strategy == strategy]
            successful = [item for item in selected if item.success]
            if selected:
                divisor = len(successful) or 1
                summary[str(strategy)] = {
                    "hit_rate": round(sum(item.hit for item in successful) / divisor, 4),
                    "mrr": round(sum(item.reciprocal_rank for item in successful) / divisor, 4),
                    "avg_latency_ms": round(sum(item.latency_ms for item in successful) / divisor, 2),
                    "successful_cases": float(len(successful)),
                    "failed_cases": float(len(selected) - len(successful)),
                }
        return EvaluationResponse(items=items, summary=summary)
