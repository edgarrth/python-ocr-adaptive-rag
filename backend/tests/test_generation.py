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
