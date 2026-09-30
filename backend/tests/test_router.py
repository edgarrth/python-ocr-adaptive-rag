from pe.axiz.payment_knowledge.domain.models import RetrievalStrategy
from pe.axiz.payment_knowledge.retrieval.router import AdaptiveRouter


def test_router_usa_hybrid_para_codigo_exacto() -> None:
    strategy, trace = AdaptiveRouter().route("¿Qué significa el código 05 en ISO 8583?")
    assert strategy == RetrievalStrategy.HYBRID_RAG
    assert trace["reason"] == "identificadores_exactos"


def test_router_usa_graphrag_para_relaciones() -> None:
    strategy, trace = AdaptiveRouter().route("¿Qué componentes están relacionados entre autorización y conciliación?")
    assert strategy == RetrievalStrategy.GRAPHRAG
    assert trace["reason"] == "relaciones"


def test_router_usa_vector_para_consulta_semantica() -> None:
    strategy, trace = AdaptiveRouter().route("Explica cómo funciona la tokenización de tarjetas")
    assert strategy == RetrievalStrategy.NATIVE_RAG
    assert trace["reason"] == "semantica"
