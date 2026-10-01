import uuid

import httpx
from pathlib import Path

from pe.axiz.payment_knowledge.domain.models import ParsedDocument
from pe.axiz.payment_knowledge.infrastructure.document_processing import PaymentEntityExtractor, SemanticChunker, canonical_markdown_name
from pe.axiz.payment_knowledge.retrieval.raglight_adapter import RagLightAdapter


def test_chunk_id_es_uuid_aceptado_por_qdrant() -> None:
    doc = ParsedDocument(
        document_id="doc-uuid",
        source_path=Path("x.md"),
        source_name="x.md",
        title="Prueba",
        markdown="# Pago\n\nAutorización con adquirente y conciliación.",
        processor="text",
    )
    chunk = SemanticChunker(PaymentEntityExtractor()).split(doc)[0]
    assert str(uuid.UUID(chunk.id)) == chunk.id


def test_raglight_recupera_nombre_original_desde_canonico() -> None:
    source = "/tmp/123456__authorization_codes.md"
    assert RagLightAdapter._source_name(source) == "authorization_codes.md"


def test_nombre_canonico_no_duplica_extension_md() -> None:
    assert canonical_markdown_name("abc123", "authorization_codes.md") == "abc123__authorization_codes.md"
    assert canonical_markdown_name("abc123", "manual.pdf") == "abc123__manual.md"


def test_raglight_error_http_incluye_detail() -> None:
    response = httpx.Response(
        500,
        json={"detail": "RateLimitError: token ausente"},
        request=httpx.Request("POST", "http://raglight/index"),
    )
    try:
        RagLightAdapter._raise_for_status(response, "index")
    except RuntimeError as exc:
        assert "HTTP 500" in str(exc)
        assert "RateLimitError: token ausente" in str(exc)
    else:
        raise AssertionError("Debió propagar el detalle devuelto por RAGLight")
