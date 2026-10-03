from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pe.axiz.payment_knowledge.application.services import IngestionService
from pe.axiz.payment_knowledge.domain.models import JobResponse, JobStatus, OcrPolicy


class IngestionJobService:
    """Cola asíncrona de ingesta para evitar requests HTTP de varios minutos."""

    def __init__(
        self,
        ingestion: "IngestionService",
        jobs_dir: Path,
        *,
        workers: int = 1,
        queue_size: int = 32,
    ) -> None:
        self.ingestion = ingestion
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.queue: asyncio.Queue[tuple[str, Path, OcrPolicy | None, bool, bool]] = asyncio.Queue(
            maxsize=max(1, queue_size)
        )
        self.worker_count = max(1, workers)
        self._workers: list[asyncio.Task[None]] = []
        self._jobs: dict[str, JobResponse] = {}

    async def start(self) -> None:
        if self._workers:
            return
        self._load_snapshots()
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"ingestion-worker-{index}")
            for index in range(self.worker_count)
        ]

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(
        self,
        source_path: Path,
        source_name: str,
        *,
        ocr_policy: OcrPolicy | None,
        index: bool,
        index_external: bool,
    ) -> JobResponse:
        job_id = uuid.uuid4().hex
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        persisted = job_dir / Path(source_name).name
        shutil.copy2(source_path, persisted)
        now = self._now()
        job = JobResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            stage="queued",
            progress=0,
            source=Path(source_name).name,
            created_at=now,
            updated_at=now,
        )
        self._jobs[job_id] = job
        self._persist(job)
        try:
            self.queue.put_nowait((job_id, persisted, ocr_policy, index, index_external))
        except asyncio.QueueFull:
            self._jobs.pop(job_id, None)
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        return job.model_copy(deep=True)

    def get(self, job_id: str) -> JobResponse | None:
        job = self._jobs.get(job_id)
        if job is not None:
            return job.model_copy(deep=True)
        snapshot = self._read_snapshot(job_id)
        if snapshot is not None:
            self._jobs[job_id] = snapshot
            return snapshot.model_copy(deep=True)
        return None

    def list(self, limit: int = 50) -> list[JobResponse]:
        jobs = sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)
        return [job.model_copy(deep=True) for job in jobs[: max(1, min(limit, 200))]]

    async def _worker(self, _: int) -> None:
        while True:
            job_id, path, ocr_policy, index, index_external = await self.queue.get()
            try:
                self._update(job_id, status=JobStatus.RUNNING, stage="starting", progress=5)

                def progress(stage: str, value: int) -> None:
                    self._update(job_id, status=JobStatus.RUNNING, stage=stage, progress=value)

                result = await self.ingestion.ingest_path(
                    path,
                    ocr_policy=ocr_policy,
                    index=index,
                    index_external=index_external,
                    progress_callback=progress,
                )
                self._update(
                    job_id,
                    status=JobStatus.SUCCEEDED,
                    stage="completed",
                    progress=100,
                    result=result,
                    error=None,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._update(
                    job_id,
                    status=JobStatus.FAILED,
                    stage="failed",
                    progress=100,
                    error=str(exc),
                )
            finally:
                self.queue.task_done()

    def _update(self, job_id: str, **changes: object) -> None:
        current = self._jobs[job_id]
        payload = current.model_dump()
        payload.update(changes)
        payload["updated_at"] = self._now()
        job = JobResponse.model_validate(payload)
        self._jobs[job_id] = job
        self._persist(job)

    def _persist(self, job: JobResponse) -> None:
        path = self.jobs_dir / job.job_id / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def _read_snapshot(self, job_id: str) -> JobResponse | None:
        path = self.jobs_dir / job_id / "job.json"
        if not path.exists():
            return None
        try:
            return JobResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _load_snapshots(self) -> None:
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = JobResponse.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                job = job.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "stage": "interrupted",
                        "progress": 100,
                        "updated_at": self._now(),
                        "error": "El backend se reinició antes de terminar este job.",
                    }
                )
                self._persist(job)
            self._jobs[job.job_id] = job

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
