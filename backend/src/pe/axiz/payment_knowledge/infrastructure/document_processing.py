from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import DocumentChunk, OcrPolicy, ParsedDocument


def canonical_markdown_name(document_id: str, source_name: str) -> str:
    """Construye un nombre canónico Markdown sin duplicar la extensión del origen."""
    return f"{document_id}__{Path(source_name).stem}.md"


class GlmOcrAdapter:
    """Encapsula GLM-OCR MaaS o self-hosted con errores observables."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def ping(self) -> bool:
        mode = self.settings.glm_ocr_mode.lower()
        if mode == "maas":
            return bool(self.settings.zhipu_api_key.strip())
        try:
            response = httpx.get(self._selfhosted_models_url(), timeout=3.0)
            return response.status_code == 200
        except Exception:
            return False

    def parse(self, path: Path) -> str:
        mode = self.settings.glm_ocr_mode.lower()
        if mode not in {"maas", "selfhosted"}:
            raise RuntimeError(f"GLM_OCR_MODE no soportado: {self.settings.glm_ocr_mode}")
        if mode == "maas":
            return self._parse_maas(path)
        return self._parse_selfhosted(path)

    def _parse_selfhosted(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf_selfhosted(path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return self._recognize_image_bytes(path.read_bytes(), mime_type)

    def _parse_pdf_selfhosted(self, path: Path) -> str:
        import pymupdf

        pages: list[str] = []
        with pymupdf.open(path) as document:
            for index, page in enumerate(document):
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), alpha=False)
                text = self._recognize_image_bytes(pixmap.tobytes("png"), "image/png")
                pages.append(f"## Página {index + 1}\n\n{text.strip()}")
        if not pages:
            raise RuntimeError("GLM-OCR no pudo rasterizar páginas del PDF")
        return "\n\n".join(pages)

    def _recognize_image_bytes(self, image: bytes, mime_type: str) -> str:
        data_url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
        payload = {
            "model": self.settings.glm_ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Text Recognition:"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": self.settings.glm_ocr_max_tokens,
        }
        try:
            response = httpx.post(
                self.settings.glm_ocr_api_url,
                json=payload,
                timeout=float(self.settings.glm_ocr_request_timeout_seconds),
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            detail = ""
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f" - {exc.response.text[:800]}"
            raise RuntimeError(
                f"GLM-OCR self-hosted no pudo inferir mediante "
                f"{self.settings.glm_ocr_api_url}: {exc}{detail}"
            ) from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Respuesta inesperada de GLM-OCR/vLLM: {body}") from exc
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        text = str(content).strip()
        if not text:
            raise RuntimeError("GLM-OCR/vLLM devolvió contenido vacío")
        return text

    def _parse_maas(self, path: Path) -> str:
        if not self.settings.zhipu_api_key:
            raise RuntimeError("ZHIPU_API_KEY es requerido para ejecutar GLM-OCR en modo MaaS")
        from glmocr import GlmOcr

        kwargs: dict[str, object] = {
            "mode": "maas",
            "model": self.settings.glm_ocr_model,
            "api_key": self.settings.zhipu_api_key,
        }
        if self.settings.glm_ocr_api_url:
            kwargs["api_url"] = self.settings.glm_ocr_api_url
        try:
            with GlmOcr(**kwargs) as parser:
                result = parser.parse(str(path))
        except Exception as exc:
            message = str(exc)
            if '"code":"1113"' in message or "余额不足" in message:
                raise RuntimeError(
                    "GLM-OCR MaaS rechazó la solicitud por saldo/paquete insuficiente (código 1113)"
                ) from exc
            raise RuntimeError(f"GLM-OCR MaaS falló: {message}") from exc

        data = result.to_dict()
        text = (
            getattr(result, "markdown_result", None)
            or data.get("markdown_result")
            or data.get("md_results")
            or data.get("text")
            or data.get("markdown")
            or data.get("content")
        )
        if not text:
            raise RuntimeError("GLM-OCR no devolvió contenido textual")
        return str(text)

    def _selfhosted_models_url(self) -> str:
        parsed = urlparse(self.settings.glm_ocr_api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise RuntimeError("GLM_OCR_API_URL self-hosted inválido")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return f"{parsed.scheme}://{parsed.hostname}:{port}/v1/models"


class DocumentProcessor:
    """Procesa con Docling y permite seleccionar explícitamente GLM-OCR."""

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    TEXT_SUFFIXES = {".md", ".txt"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ocr = GlmOcrAdapter(settings)

    def parse(self, path: Path, ocr_policy: OcrPolicy | str | None = None) -> ParsedDocument:
        document_id = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
        suffix = path.suffix.lower()
        policy = self._resolve_policy(ocr_policy)

        if suffix in self.TEXT_SUFFIXES:
            if policy == OcrPolicy.GLM:
                raise ValueError("ocr=glm solo aplica a PDF o imágenes, no a archivos de texto")
            markdown = path.read_text(encoding="utf-8")
            processor = "text"
            ocr_used = False
        elif policy == OcrPolicy.GLM or (policy == OcrPolicy.AUTO and suffix in self.IMAGE_SUFFIXES):
            # Las imágenes son OCR puro: en auto evitamos ejecutar Docling antes de GLM-OCR.
            markdown = self.ocr.parse(path)
            processor = "glm-ocr"
            ocr_used = True
        else:
            markdown = self._docling(path)
            processor = "docling"
            ocr_used = False
            should_fallback_to_glm = (
                policy == OcrPolicy.AUTO
                and len(markdown.strip()) < self.settings.ocr_min_text_chars
            )
            if should_fallback_to_glm:
                markdown = self.ocr.parse(path)
                processor = "docling+glm-ocr"
                ocr_used = True

        title = self._title(markdown, path.stem)
        return ParsedDocument(
            document_id=document_id,
            source_path=path,
            source_name=path.name,
            title=title,
            markdown=markdown,
            processor=processor,
            ocr_used=ocr_used,
        )

    def _resolve_policy(self, policy: OcrPolicy | str | None) -> OcrPolicy:
        raw = policy or self.settings.ocr_policy
        try:
            return raw if isinstance(raw, OcrPolicy) else OcrPolicy(str(raw).lower())
        except ValueError as exc:
            raise ValueError("OCR policy debe ser auto, glm o docling") from exc

    @staticmethod
    def _docling(path: Path) -> str:
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(str(path))
        return result.document.export_to_markdown()

    @staticmethod
    def _title(markdown: str, fallback: str) -> str:
        for line in markdown.splitlines():
            line = line.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip() or fallback
        return fallback.replace("_", " ").title()


class PaymentEntityExtractor:
    """Extrae entidades útiles para el grafo de conocimiento de pagos."""

    TERMS = {
        "autorización",
        "autorizacion",
        "adquirente",
        "emisor",
        "gateway",
        "conciliación",
        "conciliacion",
        "liquidación",
        "liquidacion",
        "chargeback",
        "tokenización",
        "tokenizacion",
        "idempotencia",
        "timeout",
        "iso 8583",
        "pci dss",
        "pan",
        "cvv",
        "3ds",
        "fraude",
        "reintento",
    }

    def extract(self, text: str) -> list[str]:
        lowered = text.lower()
        entities = {term.upper() for term in self.TERMS if term in lowered}
        entities.update(re.findall(r"\b(?:RC|ISO|PAY|TXN|ERR)[-_ ]?[A-Z0-9]{2,12}\b", text.upper()))
        entities.update(re.findall(r"\b(?:0[1-9]|1[0-9]|2[0-9]|5[0-9]|9[0-9])\b", text))
        return sorted(entities)


class SemanticChunker:
    """Fragmenta por bloques manteniendo encabezados y solape controlado."""

    def __init__(self, extractor: PaymentEntityExtractor, target_chars: int = 1100, overlap_chars: int = 180) -> None:
        self.extractor = extractor
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def split(self, document: ParsedDocument) -> list[DocumentChunk]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", document.markdown) if block.strip()]
        chunks: list[str] = []
        current = ""
        for block in blocks:
            candidate = f"{current}\n\n{block}".strip()
            if current and len(candidate) > self.target_chars:
                chunks.append(current)
                tail = current[-self.overlap_chars :]
                current = f"{tail}\n\n{block}".strip()
            else:
                current = candidate
        if current:
            chunks.append(current)

        result: list[DocumentChunk] = []
        for index, text in enumerate(chunks):
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pe.axiz:{document.document_id}:{index}"))
            result.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document.document_id,
                    source=document.source_name,
                    title=document.title,
                    text=text,
                    position=index,
                    entities=self.extractor.extract(text),
                    metadata={"processor": document.processor},
                )
            )
        return result
