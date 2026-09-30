from __future__ import annotations

import math
import re
from collections import Counter

from pe.axiz.payment_knowledge.domain.models import ContextItem
from pe.axiz.payment_knowledge.infrastructure.qdrant_store import QdrantStore


class NativeRagRetriever:
    def __init__(self, store: QdrantStore) -> None:
        self.store = store

    def retrieve(self, question: str, top_k: int) -> list[ContextItem]:
        contexts = self.store.search(question, top_k)
        for item in contexts:
            item.strategy = "native_rag"
        return contexts


class HybridRagRetriever:
    """Combina ranking vectorial y lexical con Reciprocal Rank Fusion."""

    def __init__(self, store: QdrantStore) -> None:
        self.store = store

    def retrieve(self, question: str, top_k: int) -> list[ContextItem]:
        dense = self.store.search(question, max(top_k * 4, 20))
        corpus = self.store.all_contexts()
        lexical = self._lexical(question, corpus)
        dense_rank = {item.id: rank for rank, item in enumerate(dense, start=1)}
        lexical_rank = {item.id: rank for rank, item in enumerate(lexical, start=1)}
        by_id = {item.id: item for item in dense + lexical}
        scores: dict[str, float] = {}
        for item_id in by_id:
            score = 0.0
            if item_id in dense_rank:
                score += 1.0 / (60 + dense_rank[item_id])
            if item_id in lexical_rank:
                score += 1.0 / (60 + lexical_rank[item_id])
            scores[item_id] = score
        ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
        result = []
        for item_id in ordered:
            item = by_id[item_id].model_copy(deep=True)
            item.score = scores[item_id]
            item.strategy = "hybrid_rag"
            result.append(item)
        return result

    def _lexical(self, question: str, corpus: list[ContextItem]) -> list[ContextItem]:
        query_terms = self._tokens(question)
        if not query_terms:
            return []
        docs = [self._tokens(item.text) for item in corpus]
        df = Counter(term for doc in docs for term in set(doc))
        total = max(len(docs), 1)
        scored: list[tuple[float, ContextItem]] = []
        for item, tokens in zip(corpus, docs, strict=True):
            tf = Counter(tokens)
            score = 0.0
            for term in query_terms:
                idf = math.log(1 + (total - df[term] + 0.5) / (df[term] + 0.5))
                score += tf[term] * idf
            if score > 0:
                clone = item.model_copy(deep=True)
                clone.score = score
                scored.append((score, clone))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9áéíóúñ_-]{2,}", text.lower())
