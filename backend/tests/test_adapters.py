import uuid

import httpx
from pathlib import Path

from pe.axiz.payment_knowledge.config import Settings
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


def test_raglight_atribuye_fuente_por_contenido_cuando_framework_no_envia_source(tmp_path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "abc123__idempotency.md").write_text(
        "# Idempotencia\n\nCada solicitud de pago usa una clave estable. "
        "Un timeout recupera el resultado original para evitar cobros duplicados.",
        encoding="utf-8",
    )
    adapter = RagLightAdapter(
        Settings(raglight_service_url="http://raglight"),
        canonical,
    )

    class FakeResponse:
        is_error = False

        @staticmethod
        def json():
            return [
                {
                    "id": "r1",
                    "source": "raglight",
                    "text": "Cada solicitud de pago usa una clave estable. Un timeout recupera el resultado original para evitar cobros duplicados.",
                    "score": 0.99,
                }
            ]

    class FakeClient:
        @staticmethod
        def post(*_args, **_kwargs):
            return FakeResponse()

        @staticmethod
        def close():
            return None

    adapter._client = FakeClient()  # type: ignore[assignment]
    contexts = adapter.retrieve("¿Cómo evita duplicados?", 5)

    assert contexts[0].source == "idempotency.md"
    assert contexts[0].document_id == "abc123"
    assert contexts[0].metadata["source_attribution"] in {"exact_content", "token_overlap"}
