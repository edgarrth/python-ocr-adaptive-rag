from pe.axiz.payment_knowledge.evaluation.metrics import ranking_metrics


def test_metricas_ranking_mrr_ndcg_recall() -> None:
    metrics = ranking_metrics(
        ["noise.md", "expected.md", "other.md"],
        ["expected.md"],
        3,
    )
    assert metrics["hit"] == 1.0
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg"] < 1
    assert metrics["recall"] == 1.0


def test_metricas_ranking_sin_hit() -> None:
    metrics = ranking_metrics(["a.md"], ["expected.md"], 5)
    assert metrics["hit"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["ndcg"] == 0.0
    assert metrics["recall"] == 0.0
