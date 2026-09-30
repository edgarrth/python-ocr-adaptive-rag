from __future__ import annotations

from pe.axiz.payment_knowledge.domain.models import ContextItem
from pe.axiz.payment_knowledge.infrastructure.memgraph_store import MemgraphStore
from pe.axiz.payment_knowledge.retrieval.native import HybridRagRetriever


class GraphRagRetriever:
    """Fusiona evidencia vectorial/híbrida con contexto conectado en Memgraph."""

    def __init__(self, graph: MemgraphStore, hybrid: HybridRagRetriever) -> None:
        self.graph = graph
        self.hybrid = hybrid

    def retrieve(self, question: str, top_k: int) -> list[ContextItem]:
        graph_items = self.graph.graph_search(question, max(top_k * 2, 10))
        hybrid_items = self.hybrid.retrieve(question, max(top_k * 2, 10))
        result: dict[str, ContextItem] = {}
        for rank, item in enumerate(graph_items, start=1):
            clone = item.model_copy(deep=True)
            clone.score = 1.5 / (50 + rank)
            clone.strategy = "graphrag"
            result[item.id] = clone
        for rank, item in enumerate(hybrid_items, start=1):
            bonus = 1.0 / (50 + rank)
            if item.id in result:
                result[item.id].score += bonus
            else:
                clone = item.model_copy(deep=True)
                clone.score = bonus
                clone.strategy = "graphrag"
                result[item.id] = clone
        return sorted(result.values(), key=lambda item: item.score, reverse=True)[:top_k]
