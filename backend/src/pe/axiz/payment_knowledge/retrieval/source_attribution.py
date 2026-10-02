from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_REFERENCE_LINE_RE = re.compile(r"^\[(?P<ref>[^\]]+)\]\s+(?P<path>\S+)\s*$")
_GENERIC_SOURCES = {"", "raglight", "lightrag", "unknown", "document", "context"}


@dataclass(frozen=True)
class SourceAttribution:
    source: str
    document_id: str
    canonical_name: str
    score: float
    method: str


def canonical_source_name(value: str) -> str:
    """Recupera el nombre original desde <document_id>__<source>.md."""
    name = Path(value).name
    if "__" in name:
        name = name.split("__", 1)[1]
    return name


def canonical_document_id(value: str) -> str:
    name = Path(value).name
    if "__" not in name:
        return ""
    return name.split("__", 1)[0]


def is_generic_source(value: str) -> bool:
    return canonical_source_name(value).strip().lower() in _GENERIC_SOURCES


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_like = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(_TOKEN_RE.findall(ascii_like.lower()))


def _token_set(value: str) -> set[str]:
    return set(normalize_text(value).split())


def _document_candidates(canonical_dir: Path) -> list[tuple[Path, str, str, str, set[str]]]:
    candidates: list[tuple[Path, str, str, str, set[str]]] = []
    for path in sorted(canonical_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        source = canonical_source_name(path.name)
        document_id = canonical_document_id(path.name)
        normalized = normalize_text(text)
        candidates.append((path, source, document_id, normalized, set(normalized.split())))
    return candidates


def infer_source_from_text(text: str, canonical_dir: Path) -> SourceAttribution | None:
    """Atribuye un fragmento al Markdown canónico usando contenido, no nombres hardcodeados."""
    normalized_text = normalize_text(text)
    query_tokens = set(normalized_text.split())
    if not query_tokens:
        return None

    best: SourceAttribution | None = None
    for path, source, document_id, normalized_doc, doc_tokens in _document_candidates(canonical_dir):
        if normalized_text and normalized_text in normalized_doc:
            score = 2.0
            method = "exact_content"
        else:
            overlap = len(query_tokens & doc_tokens)
            # El coeficiente de solapamiento favorece chunks que provienen de un documento
            # aunque el documento completo contenga mucho más vocabulario.
            score = overlap / max(1, len(query_tokens))
            method = "token_overlap"

        candidate = SourceAttribution(
            source=source,
            document_id=document_id,
            canonical_name=path.name,
            score=score,
            method=method,
        )
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None:
        return None
    # Evita atribuir por coincidencias débiles de palabras genéricas.
    if best.method != "exact_content" and best.score < 0.20:
        return None
    return best


def parse_lightrag_context(
    text: str,
    canonical_dir: Path,
    top_k: int,
) -> list[dict[str, object]]:
    """Convierte el contexto textual de LightRAG en evidencia por documento fuente."""
    references: dict[str, str] = {}
    chunks: list[tuple[str, str]] = []

    in_reference_list = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("Reference Document List"):
            in_reference_list = True
            continue

        if in_reference_list:
            match = _REFERENCE_LINE_RE.match(line)
            if match:
                references[match.group("ref")] = match.group("path").strip("`")
                continue

        if not (line.startswith("{") and '"reference_id"' in line and '"content"' in line):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        reference_id = str(payload.get("reference_id", "")).strip()
        content = str(payload.get("content", "")).strip()
        if reference_id and content:
            chunks.append((reference_id, content))

    # Agrupa chunks del mismo documento preservando el orden de aparición de LightRAG.
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for reference_id, content in chunks:
        if reference_id not in grouped:
            grouped[reference_id] = []
            order.append(reference_id)
        grouped[reference_id].append(content)

    items: list[dict[str, object]] = []
    for rank, reference_id in enumerate(order, start=1):
        content = "\n\n".join(grouped[reference_id])
        canonical_name = references.get(reference_id, "")
        source = canonical_source_name(canonical_name) if canonical_name else ""
        document_id = canonical_document_id(canonical_name) if canonical_name else ""
        method = "reference_document_list" if source else "content_inference"
        attribution_score = 1.0

        if not source:
            inferred = infer_source_from_text(content, canonical_dir)
            if inferred is not None:
                source = inferred.source
                document_id = inferred.document_id
                canonical_name = inferred.canonical_name
                attribution_score = inferred.score
                method = inferred.method

        if not source:
            source = "LightRAG"

        items.append(
            {
                "id": f"lightrag-{reference_id}",
                "document_id": document_id,
                "source": source,
                "title": Path(source).stem if source != "LightRAG" else "Contexto híbrido de LightRAG",
                "text": content,
                "score": 1.0 / rank,
                "metadata": {
                    "mode": "hybrid",
                    "reference_id": reference_id,
                    "canonical_name": canonical_name,
                    "source_attribution": method,
                    "source_attribution_score": attribution_score,
                },
            }
        )
        if len(items) >= top_k:
            break

    return items
