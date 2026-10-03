import pytest

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import ContextItem
from pe.axiz.payment_knowledge.generation.llm import AnswerGenerator


def test_generador_extractivo_no_inventa_sin_contexto() -> None:
    generator = AnswerGenerator(Settings(llm_provider="extractive"))
    answer = generator.generate("¿Qué ocurrió?", [])
    assert "No encontré evidencia suficiente" in answer


def test_generador_extractivo_incluye_fuente() -> None:
    generator = AnswerGenerator(Settings(llm_provider="extractive"))
    context = ContextItem(
        id="c1",
        document_id="d1",
        source="reconciliation.md",
        title="Conciliación",
        text="La conciliación compara autorizaciones y capturas.",
        score=1.0,
        strategy="native_rag",
    )
    answer = generator.generate("¿Qué compara la conciliación?", [context])
    assert "reconciliation.md" in answer


def test_openai_payload_no_fuerza_temperature(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        is_error = False

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "respuesta"}}]}

    def fake_post(url, *, json, headers, timeout):
        captured.update({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("pe.axiz.payment_knowledge.generation.llm.httpx.post", fake_post)
    generator = AnswerGenerator(
        Settings(
            llm_provider="openai",
            openai_api_key="test",
            openai_model="gpt-5-mini",
        )
    )
    context = ContextItem(
        id="c1",
        document_id="d1",
        source="idempotency.md",
        title="Idempotencia",
        text="Una clave estable evita duplicados.",
        score=1.0,
        strategy="native_rag",
    )

    assert generator.generate("¿Cómo evita duplicados?", [context]) == "respuesta"
    assert "temperature" not in captured["json"]


def test_openai_error_incluye_mensaje_del_proveedor() -> None:
    import httpx

    response = httpx.Response(
        400,
        json={"error": {"message": "Unsupported value: temperature"}},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    try:
        AnswerGenerator._raise_for_status(response)
    except RuntimeError as exc:
        assert "HTTP 400" in str(exc)
        assert "Unsupported value: temperature" in str(exc)
    else:
        raise AssertionError("Debió propagar el mensaje real del proveedor")


@pytest.mark.asyncio
async def test_generador_extractivo_stream_entrega_deltas() -> None:
    generator = AnswerGenerator(Settings(llm_provider="extractive"))
    context = ContextItem(
        id="c1",
        document_id="d1",
        source="authorization_codes.md",
        title="Autorización",
        text="El código 05 significa Do not honor.",
        score=1.0,
        strategy="native_rag",
    )
    parts = [part async for part in generator.stream("¿Qué significa 05?", [context])]
    assert len(parts) >= 2
    assert "authorization_codes.md" in "".join(parts)
