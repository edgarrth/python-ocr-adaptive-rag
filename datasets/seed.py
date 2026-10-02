from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Any


def _create_container() -> Any:
    # Import diferido: permite validar el CLI sin cargar todos los drivers de IA.
    from pe.axiz.payment_knowledge.container import AppContainer

    return AppContainer()


async def _wait_for_infrastructure(
    container: Any,
    wait_seconds: int,
    retry_interval: float,
) -> None:
    """Espera Qdrant, Memgraph y RAGLight sin ocupar el backend HTTP."""
    deadline = time.monotonic() + wait_seconds
    last_state: dict[str, bool] = {}

    print(
        f"[seed] Esperando Qdrant/Memgraph/RAGLight hasta {wait_seconds}s...",
        flush=True,
    )

    while time.monotonic() < deadline:
        qdrant_ok = await asyncio.to_thread(container.qdrant.ping)
        memgraph_ok = await asyncio.to_thread(container.memgraph.ping)
        raglight_ok = (
            True
            if not container.raglight.available
            else await asyncio.to_thread(container.raglight.ping)
        )
        last_state = {
            "qdrant": qdrant_ok,
            "memgraph": memgraph_ok,
            "raglight": raglight_ok,
        }
        if all(last_state.values()):
            print(
                "[seed] Qdrant, Memgraph y RAGLight disponibles. "
                "El backend HTTP no participa en la precarga.",
                flush=True,
            )
            return
        await asyncio.sleep(retry_interval)

    raise RuntimeError(
        "No se pudo disponer de la infraestructura dentro de "
        f"{wait_seconds}s. Último estado: {last_state}"
    )


async def seed_direct(wait_seconds: int, retry_interval: float) -> list[dict[str, object]]:
    """Indexa directamente desde un contenedor aislado, sin llamar al backend."""
    container = _create_container()
    try:
        await _wait_for_infrastructure(container, wait_seconds, retry_interval)
        print(
            "[seed] Iniciando precarga directa e indexación completa en proceso aislado...",
            flush=True,
        )
        started = time.monotonic()
        results = await container.ingestion.ingest_dataset()
        elapsed = time.monotonic() - started
        print(f"[seed] Precarga completada en {elapsed:.1f}s.", flush=True)
        return [item.model_dump(mode="json") for item in results]
    finally:
        await container.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Precarga el dataset de payment processing directamente contra "
            "Qdrant/Memgraph/RAGLight/LightRAG sin bloquear el backend HTTP"
        )
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=900,
        help="Tiempo máximo para esperar Qdrant, Memgraph y RAGLight",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=3.0,
        help="Segundos entre reintentos mientras se espera la infraestructura",
    )
    args = parser.parse_args()

    payload = asyncio.run(
        seed_direct(
            max(args.wait_seconds, 1),
            max(args.retry_interval, 0.2),
        )
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
