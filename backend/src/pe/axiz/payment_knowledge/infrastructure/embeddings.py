from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable


class HashingEmbedder:
    """Embedding determinista para contingencia y pruebas sin descarga de modelos."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[a-zA-Z0-9áéíóúñ_-]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.dimension
            sign = 1.0 if digest[0] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FastEmbedEmbedder:
    """Adaptador de FastEmbed con fallback determinista si el modelo no está disponible."""

    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self._fallback = HashingEmbedder(dimension)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        try:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.model_name)
            return [list(vector) for vector in self._model.embed(items)]
        except Exception:
            return self._fallback.embed(items)
