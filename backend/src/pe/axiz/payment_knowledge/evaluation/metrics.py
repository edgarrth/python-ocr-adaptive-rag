from __future__ import annotations

import math


def ranking_metrics(sources: list[str], expected_sources: list[str], top_k: int) -> dict[str, float]:
    """Métricas deterministas de ranking para retrieval evals."""
    expected = set(expected_sources)
    if not expected:
        return {"hit": 0.0, "mrr": 0.0, "ndcg": 0.0, "recall": 0.0, "precision": 0.0}
    relevant = [1 if source in expected else 0 for source in sources[:top_k]]
    first_rank = next((index for index, value in enumerate(relevant, 1) if value), None)
    dcg = sum(value / math.log2(index + 1) for index, value in enumerate(relevant, 1))
    ideal_relevant = min(len(expected), top_k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_relevant + 1))
    unique_relevant = len({source for source in sources[:top_k] if source in expected})
    return {
        "hit": 1.0 if first_rank else 0.0,
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "recall": unique_relevant / len(expected),
        "precision": sum(relevant) / max(min(top_k, len(sources)), 1),
    }
