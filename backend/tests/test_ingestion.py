from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

# Este test solo verifica la orquestación de IngestionService. En el entorno de
# pruebas ligero no están instalados los drivers de infraestructura, así que se
# crean stubs mínimos antes de importar el servicio.
if "neo4j" not in sys.modules:
    neo4j = ModuleType("neo4j")
    neo4j.GraphDatabase = object()  # type: ignore[attr-defined]
    sys.modules["neo4j"] = neo4j
if "qdrant_client" not in sys.modules:
    qdrant_client = ModuleType("qdrant_client")
    qdrant_client.QdrantClient = object  # type: ignore[attr-defined]
    qdrant_client.models = SimpleNamespace()  # type: ignore[attr-defined]
    sys.modules["qdrant_client"] = qdrant_client

from pe.axiz.payment_knowledge.application.services import IngestionService
from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import DocumentChunk, OcrPolicy


class _Processor:
    def parse(self, path: Path, ocr_policy: OcrPolicy | None = None):
        assert ocr_policy == OcrPolicy.GLM
        return SimpleNamespace(
            document_id="doc-1",
            source_name=path.name,
            title="Prueba OCR",
            markdown="# Prueba OCR\n\nPAY-42 idempotencia",
            processor="glm-ocr",
            ocr_used=True,
        )


class _Chunker:
    def split(self, document):
        return [
            DocumentChunk(
                id="chunk-1",
                document_id=document.document_id,
                source=document.source_name,
                title=document.title,
                text=document.markdown,
                position=0,
                entities=["IDEMPOTENCIA"],
            )
        ]


class _NeverCalled:
    available = True

    def __getattr__(self, name):
        raise AssertionError(f"No debe invocarse {name} cuando index=false")


class _LightNeverCalled(_NeverCalled):
    async def index(self, _path=None):
        raise AssertionError("LightRAG no debe invocarse cuando index=false")


class _QdrantRecorder:
    def __init__(self) -> None:
        self.calls = 0
        self.chunks = 0

    def document_exists(self, _document_id: str) -> bool:
        return self.chunks > 0

    def replace_document(self, chunks: list[DocumentChunk]) -> None:
        self.calls += 1
        self.chunks = len(chunks)

    def count_document(self, _document_id: str) -> int:
        return self.chunks


class _MemgraphRecorder:
    def __init__(self) -> None:
        self.calls = 0
        self.chunks = 0

    def document_exists(self, _document_id: str) -> bool:
        return self.chunks > 0

    def replace_document(self, chunks: list[DocumentChunk]) -> None:
        self.calls += 1
        self.chunks = len(chunks)

    def count_document_chunks(self, _document_id: str) -> int:
        return self.chunks


class _ExternalNeverCalled:
    available = True

    def __getattr__(self, name):
        raise AssertionError(f"El índice externo no debe invocarse: {name}")


class _ExternalLightNeverCalled(_ExternalNeverCalled):
    async def index(self, _path=None):
        raise AssertionError("LightRAG debe omitirse con index_external=false")

    async def reset(self):
        raise AssertionError("LightRAG debe omitirse con index_external=false")


def _service(tmp_path: Path) -> tuple[IngestionService, _QdrantRecorder, _MemgraphRecorder]:
    qdrant = _QdrantRecorder()
    memgraph = _MemgraphRecorder()
    service = IngestionService(
        settings=Settings(),
        processor=_Processor(),  # type: ignore[arg-type]
        chunker=_Chunker(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        memgraph=memgraph,  # type: ignore[arg-type]
        raglight=_ExternalNeverCalled(),  # type: ignore[arg-type]
        lightrag=_ExternalLightNeverCalled(),  # type: ignore[arg-type]
        canonical_dir=tmp_path / "canonical",
    )
    return service, qdrant, memgraph


def test_ingest_index_false_valida_ocr_sin_persistir(tmp_path: Path) -> None:
    sample = tmp_path / "scan.jpg"
    sample.write_bytes(b"fake")
    canonical = tmp_path / "canonical"

    service = IngestionService(
        settings=Settings(),
        processor=_Processor(),  # type: ignore[arg-type]
        chunker=_Chunker(),  # type: ignore[arg-type]
        qdrant=_NeverCalled(),  # type: ignore[arg-type]
        memgraph=_NeverCalled(),  # type: ignore[arg-type]
        raglight=_NeverCalled(),  # type: ignore[arg-type]
        lightrag=_LightNeverCalled(),  # type: ignore[arg-type]
        canonical_dir=canonical,
    )

    result = asyncio.run(
        service.ingest_path(
            sample,
            ocr_policy=OcrPolicy.GLM,
            index=False,
            index_external=False,
        )
    )

    assert result.processor == "glm-ocr"
    assert result.ocr_used is True
    assert result.indexed is False
    assert result.chunks == 1
    assert result.raglight_indexed is False
    assert result.lightrag_indexed is False
    assert result.replaced_existing is False
    assert result.idempotency_key == "doc-1"
    assert result.timings_ms["parse_ocr"] >= 0
    assert result.timings_ms["total"] >= 0
    assert list(canonical.glob("*.md")) == []


def test_ingest_index_external_false_persiste_solo_indices_base(tmp_path: Path) -> None:
    sample = tmp_path / "scan.jpg"
    sample.write_bytes(b"fake")
    service, qdrant, memgraph = _service(tmp_path)

    result = asyncio.run(
        service.ingest_path(
            sample,
            ocr_policy=OcrPolicy.GLM,
            index=True,
            index_external=False,
        )
    )

    assert result.indexed is True
    assert qdrant.calls == 1
    assert memgraph.calls == 1
    assert result.raglight_indexed is False
    assert result.lightrag_indexed is False
    assert result.replaced_existing is False
    assert len(list((tmp_path / "canonical").glob("*.md"))) == 1
    assert result.timings_ms["qdrant"] >= 0
    assert result.timings_ms["memgraph"] >= 0


def test_reingesta_mismo_documento_reemplaza_y_no_duplica(tmp_path: Path) -> None:
    sample = tmp_path / "scan.jpg"
    sample.write_bytes(b"same-content")
    service, qdrant, memgraph = _service(tmp_path)

    first = asyncio.run(
        service.ingest_path(sample, OcrPolicy.GLM, index=True, index_external=False)
    )
    second = asyncio.run(
        service.ingest_path(sample, OcrPolicy.GLM, index=True, index_external=False)
    )
    state = service.index_state("doc-1")

    assert first.replaced_existing is False
    assert second.replaced_existing is True
    assert first.idempotency_key == second.idempotency_key == "doc-1"
    assert qdrant.calls == 2
    assert memgraph.calls == 2
    assert state.expected_chunks == 1
    assert state.qdrant_chunks == 1
    assert state.memgraph_chunks == 1
    assert state.canonical_files == 1
    assert state.consistent is True


def test_mismo_contenido_con_otro_nombre_mantiene_un_solo_canonico(tmp_path: Path) -> None:
    first_path = tmp_path / "original.jpg"
    second_path = tmp_path / "copia.jpg"
    first_path.write_bytes(b"same-content")
    second_path.write_bytes(b"same-content")
    service, _, _ = _service(tmp_path)

    asyncio.run(
        service.ingest_path(first_path, OcrPolicy.GLM, index=True, index_external=False)
    )
    asyncio.run(
        service.ingest_path(second_path, OcrPolicy.GLM, index=True, index_external=False)
    )

    canonical = list((tmp_path / "canonical").glob("doc-1__*.md"))
    assert len(canonical) == 1
    assert canonical[0].name == "doc-1__copia.md"


def test_index_state_detecta_duplicados_aunque_ambos_stores_tengan_el_mismo_conteo(tmp_path: Path) -> None:
    sample = tmp_path / "scan.jpg"
    sample.write_bytes(b"same-content")
    service, qdrant, memgraph = _service(tmp_path)

    asyncio.run(
        service.ingest_path(sample, OcrPolicy.GLM, index=True, index_external=False)
    )
    qdrant.chunks = 2
    memgraph.chunks = 2

    state = service.index_state("doc-1")

    assert state.expected_chunks == 1
    assert state.qdrant_chunks == 2
    assert state.memgraph_chunks == 2
    assert state.consistent is False
