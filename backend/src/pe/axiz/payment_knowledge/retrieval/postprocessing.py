from __future__ import annotations

import math
import re
from dataclasses import dataclass

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import ContextItem
from pe.axiz.payment_knowledge.infrastructure.embeddings import FastEmbedEmbedder


@dataclass
class PostProcessTrace:
    candidates: int
    duplicates_removed: int
    reranked: bool
    deduplicated: bool


class ContextPostProcessor:
    """Deduplicación semántica + reranking común para todos los motores RAG."""

    def __init__(self, embedder: FastEmbedEmbedder, settings: Settings) -> None:
        self.embedder = embedder
        self.settings = settings

    def process(
        self,
        question: str,
        contexts: list[ContextItem],
        top_k: int,
        *,
        rerank: bool = True,
        deduplicate: bool = True,
    ) -> tuple[list[ContextItem], PostProcessTrace]:
        if not contexts:
            return [], PostProcessTrace(0, 0, rerank, deduplicate)

        clones = [item.model_copy(deep=True) for item in contexts]
        query_vector, *context_vectors = self.embedder.embed([question, *[item.text for item in clones]])
        query_terms = self._tokens(question)
        query_codes = self._codes(question)
        max_original = max((abs(float(item.score)) for item in clones), default=1.0) or 1.0

        scored: list[tuple[float, int, ContextItem, list[float]]] = []
        for rank, (item, vector) in enumerate(zip(clones, context_vectors, strict=True), start=1):
            semantic = self._cosine(query_vector, vector)
            lexical = self._lexical_overlap(query_terms, self._tokens(item.text))
            entity_boost = self._entity_boost(query_codes, item)
            original = max(0.0, min(abs(float(item.score)) / max_original, 1.0))
            rerank_score = (
                self.settings.rerank_semantic_weight * semantic
                + self.settings.rerank_lexical_weight * lexical
                + self.settings.rerank_original_weight * original
                + self.settings.rerank_entity_weight * entity_boost
            )
            item.metadata = {
                **item.metadata,
                "original_rank": rank,
                "original_score": float(item.score),
                "semantic_score": round(semantic, 6),
                "lexical_score": round(lexical, 6),
                "entity_boost": round(entity_boost, 6),
                "rerank_score": round(rerank_score, 6),
            }
            if rerank:
                item.score = rerank_score
            scored.append((rerank_score if rerank else -rank, rank, item, list(vector)))

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        duplicates_removed = 0
        kept: list[tuple[ContextItem, list[float]]] = []
        for _, _, item, vector in scored:
            is_duplicate = False
            if deduplicate:
                normalized = self._normalize_text(item.text)
                for existing, existing_vector in kept:
                    if normalized == self._normalize_text(existing.text):
                        is_duplicate = True
                        break
                    if self._cosine(vector, existing_vector) >= self.settings.semantic_dedup_threshold:
                        is_duplicate = True
                        break
            if is_duplicate:
                duplicates_removed += 1
                continue
            kept.append((item, vector))
            if len(kept) >= top_k:
                break

        return [item for item, _ in kept], PostProcessTrace(
            candidates=len(contexts),
            duplicates_removed=duplicates_removed,
            reranked=rerank,
            deduplicated=deduplicate,
        )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop = {
            "que", "como", "para", "por", "con", "del", "las", "los", "una", "uno",
            "sobre", "esta", "este", "qué", "cómo", "cuando", "cuando", "entre", "desde",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9áéíóúñ_-]{2,}", text.lower())
            if token not in stop
        }

    @staticmethod
    def _codes(text: str) -> set[str]:
        return set(re.findall(r"\b(?:0[0-9]|[1-9][0-9]|PAY[-_A-Z0-9]+|ISO\s*8583|CVV|PAN|PCI\s*DSS)\b", text.upper()))

    @staticmethod
    def _lexical_overlap(query: set[str], document: set[str]) -> float:
        if not query:
            return 0.0
        return len(query & document) / len(query)

    @staticmethod
    def _entity_boost(codes: set[str], item: ContextItem) -> float:
        if not codes:
            return 0.0
        haystack = f"{item.text} {' '.join(item.entities)}".upper()
        return sum(code in haystack for code in codes) / len(codes)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
        norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
        if not norm_a or not norm_b:
            return 0.0
        return max(-1.0, min(dot / (norm_a * norm_b), 1.0))
