from __future__ import annotations

import httpx

from pe.axiz.payment_knowledge.config import Settings
from pe.axiz.payment_knowledge.domain.models import ContextItem


class AnswerGenerator:
    """Genera respuestas con un proveedor OpenAI compatible o en modo extractivo local."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, question: str, contexts: list[ContextItem]) -> str:
        if self.settings.llm_provider.lower() == "openai" and self.settings.openai_api_key:
            return self._openai(question, contexts)
        return self._extractive(question, contexts)

    def _openai(self, question: str, contexts: list[ContextItem]) -> str:
        evidence = "\n\n".join(
            f"[{index}] {context.source} | {context.title}\n{context.text}"
            for index, context in enumerate(contexts, start=1)
        )
        payload = {
            "model": self.settings.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Responde en español usando solo la evidencia entregada. "
                        "Distingue hechos de inferencias y cita las fuentes con [n]."
                    ),
                },
                {"role": "user", "content": f"Pregunta: {question}\n\nEvidencia:\n{evidence}"},
            ],
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        response = httpx.post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _extractive(question: str, contexts: list[ContextItem]) -> str:
        if not contexts:
            return "No encontré evidencia suficiente en los documentos cargados para responder la consulta."
        lines = [
            f"Para la consulta \"{question}\" encontré evidencia en {len(contexts)} contexto(s).",
            "",
        ]
        for index, context in enumerate(contexts[:4], start=1):
            excerpt = " ".join(context.text.split())[:520]
            lines.append(f"[{index}] {context.source}: {excerpt}")
        lines.append("")
        lines.append("La respuesta está en modo extractivo; configura LLM_PROVIDER=openai para síntesis generativa.")
        return "\n".join(lines)
