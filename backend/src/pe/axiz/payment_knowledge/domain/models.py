from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class OcrPolicy(StrEnum):
    AUTO = "auto"
    GLM = "glm"
    DOCLING = "docling"


class RetrievalStrategy(StrEnum):
    AUTO = "auto"
    NATIVE_RAG = "native_rag"
    HYBRID_RAG = "hybrid_rag"
    RAGLIGHT = "raglight"
    GRAPHRAG = "graphrag"
    LIGHTRAG = "lightrag"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    source: str
    title: str
    text: str
    position: int
    entities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextItem(BaseModel):
    id: str
    document_id: str
    source: str
    title: str
    text: str
    score: float
    strategy: str
    entities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    strategy: RetrievalStrategy = RetrievalStrategy.AUTO
    top_k: int = Field(default=5, ge=1, le=20)
    include_trace: bool = True
    include_evidence: bool = True
    rerank: bool = True
    deduplicate: bool = True


class QueryResponse(BaseModel):
    answer: str
    requested_strategy: RetrievalStrategy
    executed_strategy: RetrievalStrategy
    contexts: list[ContextItem]
    trace: dict[str, Any]
    timings_ms: dict[str, float]


class IngestResponse(BaseModel):
    document_id: str
    source: str
    processor: str
    chunks: int
    entities: int
    ocr_used: bool
    indexed: bool = True
    raglight_indexed: bool
    lightrag_indexed: bool
    idempotency_key: str = ""
    replaced_existing: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentIndexState(BaseModel):
    document_id: str
    expected_chunks: int
    qdrant_chunks: int
    memgraph_chunks: int
    canonical_files: int
    canonical_sources: list[str] = Field(default_factory=list)
    consistent: bool


class DocumentSummary(BaseModel):
    document_id: str
    source: str
    title: str
    processor: str
    qdrant_chunks: int
    memgraph_chunks: int
    canonical_files: int
    entities: list[str] = Field(default_factory=list)
    consistent: bool


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    memgraph: bool
    raglight: bool
    lightrag: bool
    glm_ocr: bool
    hf_token_configured: bool


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: str
    progress: int = Field(ge=0, le=100)
    source: str
    created_at: str
    updated_at: str
    result: IngestResponse | None = None
    error: str | None = None


class EvaluationRequest(BaseModel):
    strategies: list[RetrievalStrategy] = Field(
        default_factory=lambda: [
            RetrievalStrategy.NATIVE_RAG,
            RetrievalStrategy.HYBRID_RAG,
            RetrievalStrategy.RAGLIGHT,
            RetrievalStrategy.GRAPHRAG,
            RetrievalStrategy.LIGHTRAG,
            RetrievalStrategy.AUTO,
        ]
    )
    top_k: int = Field(default=5, ge=1, le=20)
    compare_reranking: bool = True


class EvaluationItem(BaseModel):
    question: str
    category: str = "general"
    strategy: RetrievalStrategy
    expected_strategy: RetrievalStrategy | None = None
    expected_sources: list[str]
    retrieved_sources: list[str]
    raw_sources: list[str] = Field(default_factory=list)
    hit: bool
    reciprocal_rank: float
    raw_reciprocal_rank: float = 0.0
    ndcg_at_k: float = 0.0
    raw_ndcg_at_k: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    routing_correct: bool | None = None
    duplicates_removed: int = 0
    latency_ms: float
    success: bool = True
    error: str | None = None


class RankingComparison(BaseModel):
    strategy: RetrievalStrategy
    raw_mrr: float
    reranked_mrr: float
    delta_mrr: float
    raw_ndcg_at_k: float
    reranked_ndcg_at_k: float
    delta_ndcg_at_k: float
    hit_rate: float
    recall_at_k: float
    avg_latency_ms: float
    avg_duplicates_removed: float
    routing_accuracy: float | None = None
    successful_cases: int
    failed_cases: int


class EvaluationResponse(BaseModel):
    items: list[EvaluationItem]
    summary: dict[str, dict[str, float]]
    ranking_comparison: list[RankingComparison] = Field(default_factory=list)
    category_summary: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    dataset_cases: int = 0


class ParsedDocument(BaseModel):
    document_id: str
    source_path: Path
    source_name: str
    title: str
    markdown: str
    processor: str
    ocr_used: bool = False
