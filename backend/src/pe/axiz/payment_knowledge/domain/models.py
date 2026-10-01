from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RetrievalStrategy(StrEnum):
    AUTO = "auto"
    NATIVE_RAG = "native_rag"
    HYBRID_RAG = "hybrid_rag"
    RAGLIGHT = "raglight"
    GRAPHRAG = "graphrag"
    LIGHTRAG = "lightrag"


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
    raglight_indexed: bool
    lightrag_indexed: bool


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    memgraph: bool
    raglight: bool
    lightrag: bool
    hf_token_configured: bool


class EvaluationRequest(BaseModel):
    strategies: list[RetrievalStrategy] = Field(
        default_factory=lambda: [
            RetrievalStrategy.NATIVE_RAG,
            RetrievalStrategy.HYBRID_RAG,
            RetrievalStrategy.RAGLIGHT,
            RetrievalStrategy.GRAPHRAG,
            RetrievalStrategy.LIGHTRAG,
        ]
    )
    top_k: int = Field(default=5, ge=1, le=20)


class EvaluationItem(BaseModel):
    question: str
    strategy: RetrievalStrategy
    expected_sources: list[str]
    retrieved_sources: list[str]
    hit: bool
    reciprocal_rank: float
    latency_ms: float
    success: bool = True
    error: str | None = None


class EvaluationResponse(BaseModel):
    items: list[EvaluationItem]
    summary: dict[str, dict[str, float]]


class ParsedDocument(BaseModel):
    document_id: str
    source_path: Path
    source_name: str
    title: str
    markdown: str
    processor: str
    ocr_used: bool = False
