import uuid
from pathlib import Path

from pe.axiz.payment_knowledge.domain.models import ParsedDocument
from pe.axiz.payment_knowledge.infrastructure.document_processing import PaymentEntityExtractor, SemanticChunker
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
    source = "/tmp/123456__authorization_codes.md.md"
    assert RagLightAdapter._source_name(source) == "authorization_codes.md"
