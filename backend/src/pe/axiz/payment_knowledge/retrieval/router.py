from __future__ import annotations

import re

from pe.axiz.payment_knowledge.domain.models import RetrievalStrategy


class AdaptiveRouter:
    """Selecciona la estrategia según señales observables de la consulta."""

    GRAPH_TERMS = {
        "relacion",
        "relación",
        "relacionado",
        "relacionados",
        "depende",
        "flujo",
        "componentes",
        "entre",
        "impacta",
        "causa",
        "cadena",
    }
    EXACT_PATTERNS = [r"\b(?:05|51|91|96)\b", r"\bISO\s*8583\b", r"\bPAY[-_ ]?\d+\b", r"\bRC[-_ ]?\d+\b"]

    def route(self, question: str) -> tuple[RetrievalStrategy, dict[str, object]]:
        lowered = question.lower()
        graph_hits = sorted(term for term in self.GRAPH_TERMS if term in lowered)
        exact_hits = [pattern for pattern in self.EXACT_PATTERNS if re.search(pattern, question, flags=re.IGNORECASE)]
        if graph_hits:
            return RetrievalStrategy.GRAPHRAG, {"reason": "relaciones", "signals": graph_hits}
        if exact_hits:
            return RetrievalStrategy.HYBRID_RAG, {"reason": "identificadores_exactos", "signals": exact_hits}
        return RetrievalStrategy.NATIVE_RAG, {"reason": "semantica", "signals": []}
