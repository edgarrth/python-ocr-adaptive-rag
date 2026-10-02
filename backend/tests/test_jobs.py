from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pe.axiz.payment_knowledge.application.jobs import IngestionJobService
from pe.axiz.payment_knowledge.domain.models import IngestResponse, JobStatus


class FakeIngestion:
    async def ingest_path(self, path: Path, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback:
            callback("parse_ocr", 25)
            callback("qdrant", 70)
        await asyncio.sleep(0.01)
        return IngestResponse(
            document_id="doc-1",
            source=path.name,
            processor="text",
            chunks=1,
            entities=0,
            ocr_used=False,
            indexed=True,
            raglight_indexed=False,
            lightrag_indexed=False,
            idempotency_key="doc-1",
        )


@pytest.mark.asyncio
async def test_job_asincrono_llega_a_succeeded(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    source.write_text("# demo", encoding="utf-8")
    service = IngestionJobService(FakeIngestion(), tmp_path / "jobs", workers=1, queue_size=2)
    await service.start()
    try:
        submitted = await service.submit(
            source,
            source.name,
            ocr_policy=None,
            index=True,
            index_external=False,
        )
        assert submitted.status == JobStatus.QUEUED
        for _ in range(50):
            current = service.get(submitted.job_id)
            if current and current.status == JobStatus.SUCCEEDED:
                break
            await asyncio.sleep(0.01)
        current = service.get(submitted.job_id)
        assert current is not None
        assert current.status == JobStatus.SUCCEEDED
        assert current.progress == 100
        assert current.result is not None
    finally:
        await service.stop()
