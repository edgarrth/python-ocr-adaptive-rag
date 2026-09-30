from pathlib import Path

from pe.axiz.payment_knowledge.domain.models import ParsedDocument
from pe.axiz.payment_knowledge.infrastructure.document_processing import PaymentEntityExtractor, SemanticChunker


def test_chunker_detecta_entidades_de_pago() -> None:
    doc = ParsedDocument(
        document_id="doc-1",
        source_path=Path("x.md"),
        source_name="x.md",
        title="Prueba",
        markdown="# Pago\n\nLa autorización ISO 8583 usa código 05 y pasa por el adquirente.",
        processor="text",
    )
    chunks = SemanticChunker(PaymentEntityExtractor(), target_chars=200).split(doc)
    assert chunks
    assert any("ISO 8583" in entity for entity in chunks[0].entities)
