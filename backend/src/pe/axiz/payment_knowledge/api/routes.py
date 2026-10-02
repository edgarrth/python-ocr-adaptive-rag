from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from pe.axiz.payment_knowledge.container import get_container
from pe.axiz.payment_knowledge.domain.models import (
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IngestResponse,
    OcrPolicy,
    QueryRequest,
    QueryResponse,
)

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    container = get_container()
    qdrant_ok = container.qdrant.ping()
    memgraph_ok = container.memgraph.ping()
    raglight_ok = container.raglight.ping()
    lightrag_ok = container.lightrag.available
    glm_ocr_ok = container.ingestion.processor.ocr.ping()
    return HealthResponse(
        status=(
            "ok"
            if qdrant_ok and memgraph_ok and raglight_ok and lightrag_ok and glm_ocr_ok
            else "degraded"
        ),
        qdrant=qdrant_ok,
        memgraph=memgraph_ok,
        raglight=raglight_ok,
        lightrag=lightrag_ok,
        glm_ocr=glm_ocr_ok,
        hf_token_configured=bool(container.settings.hf_token.strip()),
    )


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    ocr: OcrPolicy | None = Query(default=None, description="auto | glm | docling"),
) -> IngestResponse:
    source_name = Path(file.filename or "document.bin").name
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / source_name
        with temp_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
        try:
            return await get_container().ingestion.ingest_path(temp_path, ocr_policy=ocr)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/datasets/seed", response_model=list[IngestResponse])
async def seed_dataset() -> list[IngestResponse]:
    try:
        return await get_container().ingestion.ingest_dataset()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        return await get_container().query.query(request)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/graph/neighborhood/{entity}")
def graph_neighborhood(entity: str) -> dict[str, list[dict[str, object]]]:
    try:
        return get_container().memgraph.neighborhood(entity)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/evaluations/run", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    try:
        return await get_container().evaluation.run(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
