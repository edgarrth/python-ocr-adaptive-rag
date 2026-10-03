from __future__ import annotations

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import ContextItem
from pe.axiz.payment_knowledge.retrieval.postprocessing import ContextPostProcessor


class FakeEmbedder:
    def embed(self, texts):
        vectors = []
        for text in texts:
            value = str(text).lower()
            if "05" in value or "do not honor" in value:
                vectors.append([1.0, 0.0])
            elif "duplicado" in value:
                vectors.append([0.99, 0.01])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def _context(identifier: str, source: str, text: str, score: float) -> ContextItem:
    return ContextItem(
        id=identifier,
        document_id=identifier,
        source=source,
        title=source,
        text=text,
        score=score,
        strategy="test",
        entities=[],
    )


def test_reranker_promueve_contexto_semanticamente_relevante() -> None:
    processor = ContextPostProcessor(FakeEmbedder(), Settings())
    raw = [
        _context("a", "irrelevant.md", "texto sobre conciliación", 0.95),
        _context("b", "authorization_codes.md", "El código 05 significa Do not honor", 0.20),
    ]

    result, trace = processor.process("¿Qué significa el código 05?", raw, 2)

    assert result[0].source == "authorization_codes.md"
    assert result[0].metadata["original_rank"] == 2
    assert trace.reranked is True


def test_deduplicacion_semantica_elimina_contextos_equivalentes() -> None:
    settings = Settings(semantic_dedup_threshold=0.94)
    processor = ContextPostProcessor(FakeEmbedder(), settings)
    raw = [
        _context("a", "a.md", "El código 05 significa Do not honor", 0.9),
        _context("b", "b.md", "Código 05: Do not honor y rechazo", 0.8),
        _context("c", "c.md", "Conciliación y settlement", 0.7),
    ]

    result, trace = processor.process("código 05", raw, 3, deduplicate=True)

    assert len(result) == 2
    assert trace.duplicates_removed == 1
