from pathlib import Path

import pytest

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import OcrPolicy
from pe.axiz.payment_knowledge.infrastructure.document_processing import DocumentProcessor, GlmOcrAdapter


def test_selfhosted_url_deriva_models() -> None:
    adapter = GlmOcrAdapter(
        Settings(glm_ocr_mode="selfhosted", glm_ocr_api_url="http://glm-ocr:8080/v1/chat/completions")
    )
    assert adapter._selfhosted_models_url() == "http://glm-ocr:8080/v1/models"


def test_selfhosted_imagen_llama_directamente_vllm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"jpeg-bytes")
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "PAY-42\nIdempotencia"}}]}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("pe.axiz.payment_knowledge.infrastructure.document_processing.httpx.post", fake_post)
    adapter = GlmOcrAdapter(
        Settings(
            glm_ocr_mode="selfhosted",
            glm_ocr_api_url="http://glm-ocr:8080/v1/chat/completions",
            glm_ocr_model="glm-ocr",
            glm_ocr_max_tokens=1024,
        )
    )

    text = adapter.parse(image)

    assert "PAY-42" in text
    assert captured["url"] == "http://glm-ocr:8080/v1/chat/completions"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "glm-ocr"
    assert payload["max_tokens"] == 1024
    content = payload["messages"][0]["content"]  # type: ignore[index]
    assert content[0]["text"] == "Text Recognition:"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


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


def test_preprocesado_ocr_reduce_imagen_grande(tmp_path: Path) -> None:
    import io
    from PIL import Image

    image = Image.new("RGB", (2200, 1600), "white")
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    adapter = GlmOcrAdapter(
        Settings(glm_ocr_image_max_side=1000, glm_ocr_image_max_pixels=800000)
    )

    prepared, mime = adapter._prepare_image_bytes(raw.getvalue(), "image/png")

    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(prepared)) as result:
        assert max(result.size) <= 1000
        assert result.size[0] * result.size[1] <= 800000
