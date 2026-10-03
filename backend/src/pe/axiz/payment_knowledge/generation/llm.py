from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

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

    async def stream(self, question: str, contexts: list[ContextItem]) -> AsyncIterator[str]:
        """Entrega deltas visibles sin exponer razonamiento interno del modelo."""
        if self.settings.llm_provider.lower() == "openai" and self.settings.openai_api_key:
            async for delta in self._openai_stream(question, contexts):
                yield delta
            return

        answer = self._extractive(question, contexts)
        chunk_size = 72
        for start in range(0, len(answer), chunk_size):
            yield answer[start : start + chunk_size]
            await asyncio.sleep(0)

    def _messages(self, question: str, contexts: list[ContextItem]) -> list[dict[str, str]]:
        evidence = "\n\n".join(
            f"[{index}] {context.source} | {context.title}\n{context.text}"
            for index, context in enumerate(contexts, start=1)
        )
        return [
            {
                "role": "system",
                "content": (
                    "Responde en español usando solo la evidencia entregada. "
                    "Distingue hechos de inferencias y cita las fuentes con [n]."
                ),
            },
            {"role": "user", "content": f"Pregunta: {question}\n\nEvidencia:\n{evidence}"},
        ]

    def _openai(self, question: str, contexts: list[ContextItem]) -> str:
        payload = {
            "model": self.settings.openai_model,
            "messages": self._messages(question, contexts),
        }
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        response = httpx.post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=90,
        )
        self._raise_for_status(response)
        return response.json()["choices"][0]["message"]["content"]

    async def _openai_stream(
        self,
        question: str,
        contexts: list[ContextItem],
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.settings.openai_model,
            "messages": self._messages(question, contexts),
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Accept": "text/event-stream",
        }
        timeout = httpx.Timeout(180.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.is_error:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"OpenAI-compatible API respondió HTTP {response.status_code}: {body}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if isinstance(delta, str) and delta:
                        yield delta

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if not response.is_error:
            return
        detail = response.text
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict) and error.get("message"):
                    detail = str(error["message"])
                elif payload.get("detail"):
                    detail = str(payload["detail"])
        except ValueError:
            pass
        raise RuntimeError(
            f"OpenAI-compatible API respondió HTTP {response.status_code}: {detail}"
        )

    @staticmethod
    def _extractive(question: str, contexts: list[ContextItem]) -> str:
        if not contexts:
            return "No encontré evidencia suficiente en los documentos cargados para responder la consulta."
        lines = [
            f'Para la consulta "{question}" encontré evidencia en {len(contexts)} contexto(s).',
            "",
        ]
        for index, context in enumerate(contexts[:4], start=1):
            excerpt = " ".join(context.text.split())[:520]
            lines.append(f"[{index}] {context.source}: {excerpt}")
        lines.append("")
        lines.append(
            "La respuesta está en modo extractivo; configura LLM_PROVIDER=openai "
            "para síntesis generativa."
        )
        return "\n".join(lines)
