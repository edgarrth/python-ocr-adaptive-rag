from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import DocumentChunk, ParsedDocument


def canonical_markdown_name(document_id: str, source_name: str) -> str:
    """Construye un nombre canónico Markdown sin duplicar la extensión del origen."""
    return f"{document_id}__{Path(source_name).stem}.md"


class GlmOcrAdapter:
    """Encapsula GLM-OCR para documentos escaneados o imágenes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def parse(self, path: Path) -> str:
        if self.settings.glm_ocr_mode.lower() == "maas" and not self.settings.zhipu_api_key:
            raise RuntimeError("ZHIPU_API_KEY es requerido para ejecutar GLM-OCR en modo MaaS")
        from glmocr import GlmOcr

        with GlmOcr(
            api_key=self.settings.zhipu_api_key or None,
            api_url=self.settings.glm_ocr_api_url or None,
            mode=self.settings.glm_ocr_mode,
        ) as parser:
            result = parser.parse(str(path))
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


class DocumentProcessor:
    """Procesa archivos con Docling y deriva a GLM-OCR cuando hace falta."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ocr = GlmOcrAdapter(settings)

    def parse(self, path: Path) -> ParsedDocument:
        document_id = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            markdown = path.read_text(encoding="utf-8")
            processor = "text"
        else:
            markdown = self._docling(path)
            processor = "docling"

        ocr_used = False
        if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"} or len(markdown.strip()) < self.settings.ocr_min_text_chars:
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
