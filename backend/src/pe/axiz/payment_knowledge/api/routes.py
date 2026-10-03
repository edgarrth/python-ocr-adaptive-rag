from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from pe.axiz.payment_knowledge.container import AppContainer, get_container
from pe.axiz.payment_knowledge.domain.models import (
    DocumentIndexState,
    DocumentSummary,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IngestResponse,
    JobResponse,
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
    index: bool = Query(
        default=True,
        description="false valida parse/OCR+chunking sin persistir en ningún índice",
    ),
    index_external: bool = Query(
        default=True,
        description="false indexa solo Qdrant/Memgraph y omite RAGLight/LightRAG",
    ),
) -> IngestResponse:
    """Ruta síncrona conservada para pruebas y automatizaciones simples."""
    source_name = Path(file.filename or "document.bin").name
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / source_name
        with temp_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
        try:
            return await get_container().ingestion.ingest_path(
                temp_path,
                ocr_policy=ocr,
                index=index,
                index_external=index_external,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/documents/jobs", response_model=JobResponse, status_code=202)
async def enqueue_document(
    file: UploadFile = File(...),
    ocr: OcrPolicy | None = Query(default=None, description="auto | glm | docling"),
    index: bool = Query(default=True),
    index_external: bool = Query(default=False),
) -> JobResponse:
    """Ingesta asíncrona: responde 202 y deja OCR/indexación en un worker."""
    source_name = Path(file.filename or "document.bin").name
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / source_name
        with temp_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
        try:
            return await get_container().jobs.submit(
                temp_path,
                source_name,
                ocr_policy=ocr,
                index=index,
                index_external=index_external,
            )
        except asyncio.QueueFull as exc:
            raise HTTPException(status_code=429, detail="La cola de ingesta está llena") from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(limit: int = Query(default=50, ge=1, le=200)) -> list[JobResponse]:
    return get_container().jobs.list(limit)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    job = get_container().jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return job


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    try:
        return get_container().ingestion.list_documents()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/documents/{document_id}/index-state", response_model=DocumentIndexState)
def document_index_state(document_id: str) -> DocumentIndexState:
    try:
        return get_container().ingestion.index_state(document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    rebuild_external: bool = Query(default=False),
) -> dict[str, object]:
    try:
        existed = await get_container().ingestion.delete_document(
            document_id,
            rebuild_external=rebuild_external,
        )
        if not existed:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        return {"document_id": document_id, "deleted": True, "external_rebuilt": rebuild_external}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/documents/{document_id}/reindex", response_model=IngestResponse)
async def reindex_document(
    document_id: str,
    rebuild_external: bool = Query(default=False),
) -> IngestResponse:
    try:
        return await get_container().ingestion.reindex_document(
            document_id,
            rebuild_external=rebuild_external,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _run_seed_in_isolated_container() -> list[IngestResponse]:
    """Ejecuta el seed en un hilo con su propio event loop y contenedor."""

    async def _run() -> list[IngestResponse]:
        container = AppContainer()
        try:
            return await container.ingestion.ingest_dataset()
        finally:
            await container.close()

    return asyncio.run(_run())


@router.post("/datasets/seed", response_model=list[IngestResponse])
async def seed_dataset() -> list[IngestResponse]:
    try:
        return await run_in_threadpool(_run_seed_in_isolated_container)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        return await get_container().query.query(request)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _sse(event: str, data: dict[str, object]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    async def events():
        try:
            async for event, data in get_container().query.stream(request):
                yield _sse(event, data)
        except Exception as exc:
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/graph/neighborhood/{entity}")
def graph_neighborhood(entity: str) -> dict[str, list[dict[str, object]]]:
    try:
        return get_container().memgraph.neighborhood(entity)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/graph/overview")
def graph_overview(limit: int = Query(default=80, ge=5, le=300)) -> dict[str, list[dict[str, object]]]:
    try:
        return get_container().memgraph.overview(limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/evaluations/run", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    try:
        return await get_container().evaluation.run(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
