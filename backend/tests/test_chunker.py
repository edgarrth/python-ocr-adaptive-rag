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


def test_chunk_id_depende_de_documento_y_posicion_no_del_texto() -> None:
    chunker = SemanticChunker(PaymentEntityExtractor(), target_chars=500)
    doc_a = ParsedDocument(
        document_id="same-doc",
        source_path=Path("scan.jpg"),
        source_name="scan.jpg",
        title="OCR",
        markdown="# OCR\n\ntexto variante A",
        processor="glm-ocr",
    )
    doc_b = doc_a.model_copy(update={"markdown": "# OCR\n\ntexto variante B"})

    chunk_a = chunker.split(doc_a)[0]
    chunk_b = chunker.split(doc_b)[0]

    assert chunk_a.id == chunk_b.id
