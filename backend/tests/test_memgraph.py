import sys
import types

neo4j = types.ModuleType("neo4j")
neo4j.GraphDatabase = object()  # type: ignore[attr-defined]
sys.modules.setdefault("neo4j", neo4j)

from pe.axiz.payment_knowledge.infrastructure.memgraph_store import MemgraphStore


def test_normalizacion_neighborhood_ignora_acentos_y_case() -> None:
    assert MemgraphStore._normalize_search_text("AUTORIZACIÓN") == "autorizacion"
    assert MemgraphStore._normalize_search_text("autorizacion") == "autorizacion"
