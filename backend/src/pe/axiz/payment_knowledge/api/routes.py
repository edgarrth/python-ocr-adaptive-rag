from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from pe.axiz.payment_knowledge.container import get_container
from pe.axiz.payment_knowledge.domain.models import EvaluationRequest, EvaluationResponse, HealthResponse, IngestResponse, QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    container = get_container()
    qdrant_ok = container.qdrant.ping()
    memgraph_ok = container.memgraph.ping()
    return HealthResponse(
        status="ok" if qdrant_ok and memgraph_ok else "degraded",
        qdrant=qdrant_ok,
        memgraph=memgraph_ok,
        raglight=container.raglight.ping(),
        lightrag=container.lightrag.available,
    )


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    suffix = Path(file.filename or "document.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = Path(temp.name)
    try:
        return await get_container().ingestion.ingest_path(temp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


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
