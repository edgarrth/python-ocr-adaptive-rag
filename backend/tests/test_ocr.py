from pathlib import Path

import pytest

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import OcrPolicy
from pe.axiz.payment_knowledge.infrastructure.document_processing import DocumentProcessor, GlmOcrAdapter


def test_selfhosted_url_deriva_host_port_y_models() -> None:
    adapter = GlmOcrAdapter(
        Settings(glm_ocr_mode="selfhosted", glm_ocr_api_url="http://glm-ocr:8080/v1/chat/completions")
    )
    assert adapter._selfhosted_host_port() == ("glm-ocr", 8080)
    assert adapter._selfhosted_models_url() == "http://glm-ocr:8080/v1/models"


def test_ocr_policy_forzada_glm_omite_docling(tmp_path: Path) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"fake-image")
    processor = DocumentProcessor(Settings(ocr_policy="auto"))
    processor._docling = lambda _path: (_ for _ in ()).throw(AssertionError("Docling no debe ejecutarse"))  # type: ignore[method-assign]
    processor.ocr.parse = lambda _path: "# OCR\n\nPAY-42 timeout idempotencia"  # type: ignore[method-assign]

    parsed = processor.parse(image, OcrPolicy.GLM)

    assert parsed.ocr_used is True
    assert parsed.processor == "glm-ocr"
    assert "PAY-42" in parsed.markdown


def test_ocr_policy_docling_no_invoca_glm(tmp_path: Path) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"fake-image")
    processor = DocumentProcessor(Settings(ocr_policy="auto"))
    processor._docling = lambda _path: "# Docling\n\ntexto extraido"  # type: ignore[method-assign]
    processor.ocr.parse = lambda _path: (_ for _ in ()).throw(AssertionError("GLM no debe ejecutarse"))  # type: ignore[method-assign]

    parsed = processor.parse(image, OcrPolicy.DOCLING)

    assert parsed.ocr_used is False
    assert parsed.processor == "docling"


def test_ocr_glm_rechaza_archivo_texto(tmp_path: Path) -> None:
    text_file = tmp_path / "x.txt"
    text_file.write_text("hola", encoding="utf-8")
    processor = DocumentProcessor(Settings())
    with pytest.raises(ValueError, match="solo aplica a PDF o imágenes"):
        processor.parse(text_file, OcrPolicy.GLM)


def test_ocr_auto_en_imagen_va_directo_a_glm(tmp_path: Path) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"fake-image")
    processor = DocumentProcessor(Settings(ocr_policy="auto"))
    processor._docling = lambda _path: (_ for _ in ()).throw(AssertionError("Docling no debe ejecutarse"))  # type: ignore[method-assign]
    processor.ocr.parse = lambda _path: "# OCR\n\ntexto desde GLM"  # type: ignore[method-assign]

    parsed = processor.parse(image)

    assert parsed.processor == "glm-ocr"
    assert parsed.ocr_used is True
