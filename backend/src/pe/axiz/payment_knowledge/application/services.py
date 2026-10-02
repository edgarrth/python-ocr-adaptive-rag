from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import (
    ContextItem,
    DocumentIndexState,
    EvaluationItem,
    EvaluationRequest,
    EvaluationResponse,
    IngestResponse,
    OcrPolicy,
    ParsedDocument,
    QueryRequest,
    QueryResponse,
    RetrievalStrategy,
)
from pe.axiz.payment_knowledge.generation.llm import AnswerGenerator
from pe.axiz.payment_knowledge.infrastructure.document_processing import DocumentProcessor, PaymentEntityExtractor, SemanticChunker, canonical_markdown_name
from pe.axiz.payment_knowledge.infrastructure.memgraph_store import MemgraphStore
from pe.axiz.payment_knowledge.infrastructure.qdrant_store import QdrantStore
from pe.axiz.payment_knowledge.retrieval.graphrag import GraphRagRetriever
from pe.axiz.payment_knowledge.retrieval.lightrag_adapter import LightRagAdapter
from pe.axiz.payment_knowledge.retrieval.native import HybridRagRetriever, NativeRagRetriever
from pe.axiz.payment_knowledge.retrieval.raglight_adapter import RagLightAdapter
from pe.axiz.payment_knowledge.retrieval.router import AdaptiveRouter


logger = logging.getLogger(__name__)


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
        self._document_locks: dict[str, asyncio.Lock] = {}
        self._external_rebuild_lock = asyncio.Lock()

    async def ingest_path(
        self,
        path: Path,
        ocr_policy: OcrPolicy | None = None,
        *,
        index: bool = True,
        index_external: bool = True,
    ) -> IngestResponse:
        document_key = await asyncio.to_thread(self._content_key, path)
        lock = self._document_locks.setdefault(document_key, asyncio.Lock())
        async with lock:
            return await self._ingest_path(
                path,
                index_base=index,
                index_raglight=index and index_external,
                index_lightrag=index and index_external,
                ocr_policy=ocr_policy,
            )

    async def _ingest_path(
        self,
        path: Path,
        *,
        index_base: bool = True,
        index_raglight: bool,
        index_lightrag: bool,
        ocr_policy: OcrPolicy | None = None,
    ) -> IngestResponse:
        timings: dict[str, float] = {}
        total_started = time.perf_counter()

        stage_started = time.perf_counter()
        logger.info("Ingesta %s: iniciando parse/OCR", path.name)
        document = await asyncio.to_thread(self.processor.parse, path, ocr_policy)
        timings["parse_ocr"] = round((time.perf_counter() - stage_started) * 1000, 2)
        logger.info(
            "Ingesta %s: parse/OCR completado con %s en %.2f ms",
            path.name,
            document.processor,
            timings["parse_ocr"],
        )

        stage_started = time.perf_counter()
        chunks = self.chunker.split(document)
        timings["chunking"] = round((time.perf_counter() - stage_started) * 1000, 2)

        raglight_indexed = False
        lightrag_indexed = False
        replaced_existing = False

        if index_base:
            stage_started = time.perf_counter()
            qdrant_existing, memgraph_existing = await asyncio.gather(
                asyncio.to_thread(self.qdrant.document_exists, document.document_id),
                asyncio.to_thread(self.memgraph.document_exists, document.document_id),
            )
            replaced_existing = qdrant_existing or memgraph_existing
            timings["idempotency_lookup"] = round(
                (time.perf_counter() - stage_started) * 1000, 2
            )

            stage_started = time.perf_counter()
            logger.info(
                "Ingesta %s: reemplazando representación de %s chunks en Qdrant",
                path.name,
                len(chunks),
            )
            await asyncio.to_thread(self.qdrant.replace_document, chunks)
            timings["qdrant"] = round((time.perf_counter() - stage_started) * 1000, 2)

            stage_started = time.perf_counter()
            logger.info("Ingesta %s: reemplazando subgrafo en Memgraph", path.name)
            await asyncio.to_thread(self.memgraph.replace_document, chunks)
            timings["memgraph"] = round((time.perf_counter() - stage_started) * 1000, 2)

            stage_started = time.perf_counter()
            canonical_path = self._write_canonical(document)
            timings["canonical"] = round((time.perf_counter() - stage_started) * 1000, 2)

            if (
                (index_raglight and self.raglight.available)
                or (index_lightrag and self.lightrag.available)
            ):
                async with self._external_rebuild_lock:
                    if index_raglight and self.raglight.available:
                        stage_started = time.perf_counter()
                        logger.info(
                            "Ingesta %s: reconstruyendo RAGLight desde el corpus canónico "
                            "para evitar duplicados",
                            path.name,
                        )
                        await asyncio.to_thread(self.raglight.reset)
                        await asyncio.to_thread(self.raglight.ready)
                        try:
                            raglight_indexed = await asyncio.wait_for(
                                asyncio.to_thread(self.raglight.index, self.canonical_dir),
                                timeout=float(self.settings.raglight_timeout_seconds) + 5.0,
                            )
                        except TimeoutError as exc:
                            raise RuntimeError(
                                f"RAGLight excedió {self.settings.raglight_timeout_seconds:.0f}s "
                                f"reconstruyendo el corpus tras {path.name}"
                            ) from exc
                        timings["raglight"] = round(
                            (time.perf_counter() - stage_started) * 1000, 2
                        )

                    if index_lightrag and self.lightrag.available:
                        stage_started = time.perf_counter()
                        logger.info(
                            "Ingesta %s: reconstruyendo LightRAG desde el corpus canónico "
                            "para evitar duplicados",
                            path.name,
                        )
                        await self.lightrag.reset()
                        lightrag_indexed = await self.lightrag.index()
                        timings["lightrag"] = round(
                            (time.perf_counter() - stage_started) * 1000, 2
                        )
        else:
            logger.info(
                "Ingesta %s: index=false, se validó parse/OCR+chunking sin persistir índices",
                path.name,
            )

        timings["total"] = round((time.perf_counter() - total_started) * 1000, 2)
        return IngestResponse(
            document_id=document.document_id,
            source=document.source_name,
            processor=document.processor,
            chunks=len(chunks),
            entities=len({entity for chunk in chunks for entity in chunk.entities}),
            ocr_used=document.ocr_used,
            indexed=index_base,
            raglight_indexed=raglight_indexed,
            lightrag_indexed=lightrag_indexed,
            idempotency_key=document.document_id,
            replaced_existing=replaced_existing,
            timings_ms=timings,
        )

    async def ingest_dataset(self) -> list[IngestResponse]:
        self._remove_legacy_canonical_files()

        if self.raglight.available:
            logger.info("Preparando RAGLight para el bootstrap")
            if not await asyncio.to_thread(self.raglight.ping):
                raise RuntimeError("RAGLight está habilitado pero su servicio no está disponible")
            await asyncio.to_thread(self.raglight.reset)
            if not await asyncio.to_thread(self.raglight.ready):
                raise RuntimeError("RAGLight no completó su preparación de embeddings/Qdrant")

        if self.lightrag.available:
            logger.info("Reiniciando el storage de LightRAG")
            await self.lightrag.reset()

        paths = [path for path in sorted(self.settings.datasets_dir.glob("*")) if path.is_file()]
        results: list[IngestResponse] = []
        for index, path in enumerate(paths, start=1):
            logger.info("Indexando documento %s/%s: %s", index, len(paths), path.name)
            result = await self._ingest_path(
                path,
                index_base=True,
                index_raglight=False,
                index_lightrag=False,
                ocr_policy=None,
            )
            results.append(result)
            logger.info(
                "Documento indexado: %s, chunks=%s, entidades=%s, reemplazado=%s, lightrag=%s",
                result.source,
                result.chunks,
                result.entities,
                result.replaced_existing,
                result.lightrag_indexed,
            )

        raglight_indexed = False
        if self.raglight.available:
            logger.info("Indexando el corpus canónico completo en RAGLight")
            raglight_indexed = await asyncio.to_thread(self.raglight.index, self.canonical_dir)
            if not raglight_indexed:
                raise RuntimeError("RAGLight no confirmó la indexación del dataset")

        lightrag_indexed = False
        if self.lightrag.available:
            logger.info("Indexando el corpus canónico completo en LightRAG")
            lightrag_indexed = await self.lightrag.index()
            if not lightrag_indexed:
                raise RuntimeError("LightRAG no confirmó la indexación del dataset")

        if raglight_indexed or lightrag_indexed:
            results = [
                result.model_copy(
                    update={
                        "raglight_indexed": raglight_indexed or result.raglight_indexed,
                        "lightrag_indexed": lightrag_indexed or result.lightrag_indexed,
                    }
                )
                for result in results
            ]
        return results

    def index_state(self, document_id: str) -> DocumentIndexState:
        qdrant_chunks = self.qdrant.count_document(document_id)
        memgraph_chunks = self.memgraph.count_document_chunks(document_id)
        canonical = sorted(self.canonical_dir.glob(f"{document_id}__*.md"))
        canonical_files = len(canonical)
        expected_chunks = 0
        if canonical_files == 1:
            canonical_path = canonical[0]
            canonical_document = ParsedDocument(
                document_id=document_id,
                source_path=canonical_path,
                source_name=canonical_path.name.split("__", 1)[-1],
                title=canonical_path.stem,
                markdown=canonical_path.read_text(encoding="utf-8"),
                processor="canonical-state",
            )
            expected_chunks = len(self.chunker.split(canonical_document))

        consistent = (
            canonical_files <= 1
            and qdrant_chunks == memgraph_chunks == expected_chunks
            and (canonical_files == 1 or expected_chunks == 0)
        )
        return DocumentIndexState(
            document_id=document_id,
            expected_chunks=expected_chunks,
            qdrant_chunks=qdrant_chunks,
            memgraph_chunks=memgraph_chunks,
            canonical_files=canonical_files,
            canonical_sources=[path.name for path in canonical],
            consistent=consistent,
        )


    @staticmethod
    def _content_key(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:24]

    def _write_canonical(self, document) -> Path:
        canonical_path = self._canonical_path(document.document_id, document.source_name)
        for previous in self.canonical_dir.glob(f"{document.document_id}__*.md"):
            if previous != canonical_path:
                previous.unlink(missing_ok=True)
        canonical_path.write_text(document.markdown, encoding="utf-8")
        return canonical_path

    def _canonical_path(self, document_id: str, source_name: str) -> Path:
        return self.canonical_dir / canonical_markdown_name(document_id, source_name)

    def _remove_legacy_canonical_files(self) -> None:
        for path in self.canonical_dir.glob("*.md.md"):
            path.unlink(missing_ok=True)
        shutil.rmtree(self.canonical_dir / "_raglight_stage", ignore_errors=True)


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
