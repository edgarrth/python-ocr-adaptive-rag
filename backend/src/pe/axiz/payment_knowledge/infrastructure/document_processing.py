from __future__ import annotations

import hashlib
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
            endpoint = self._selfhosted_models_url()
            response = httpx.get(endpoint, timeout=3.0)
            return response.status_code == 200
        except Exception:
            return False

    def parse(self, path: Path) -> str:
        mode = self.settings.glm_ocr_mode.lower()
        if mode not in {"maas", "selfhosted"}:
            raise RuntimeError(f"GLM_OCR_MODE no soportado: {self.settings.glm_ocr_mode}")
        if mode == "maas" and not self.settings.zhipu_api_key:
            raise RuntimeError("ZHIPU_API_KEY es requerido para ejecutar GLM-OCR en modo MaaS")

        from glmocr import GlmOcr

        kwargs: dict[str, object] = {
            "mode": mode,
            "model": self.settings.glm_ocr_model,
            "layout_device": self.settings.glm_ocr_layout_device,
            "_dotted": {
                "pipeline.max_workers": self.settings.glm_ocr_max_workers,
                "pipeline.ocr_api.request_timeout": self.settings.glm_ocr_request_timeout_seconds,
            },
        }
        if mode == "maas":
            kwargs["api_key"] = self.settings.zhipu_api_key
            if self.settings.glm_ocr_api_url:
                kwargs["api_url"] = self.settings.glm_ocr_api_url
        else:
            host, port = self._selfhosted_host_port()
            kwargs["ocr_api_host"] = host
            kwargs["ocr_api_port"] = port

        try:
            with GlmOcr(**kwargs) as parser:
                result = parser.parse(str(path))
        except Exception as exc:
            message = str(exc)
            if '"code":"1113"' in message or "余额不足" in message:
                raise RuntimeError(
                    "GLM-OCR MaaS rechazó la solicitud por saldo/paquete insuficiente (código 1113)"
                ) from exc
            if mode == "selfhosted":
                raise RuntimeError(
                    f"GLM-OCR self-hosted no pudo procesar el documento usando "
                    f"{self.settings.glm_ocr_api_url}: {message}"
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

    def _selfhosted_host_port(self) -> tuple[str, int]:
        parsed = urlparse(self.settings.glm_ocr_api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise RuntimeError(
                "GLM_OCR_API_URL self-hosted debe ser una URL HTTP válida, por ejemplo "
                "http://glm-ocr:8080/v1/chat/completions"
            )
        return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)

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
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pe.axiz:{document.document_id}:{index}:{text}"))
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
