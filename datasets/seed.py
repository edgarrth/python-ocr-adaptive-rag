from __future__ import annotations

import argparse
import json
import time

import httpx


def _wait_for_infrastructure(api: str, wait_seconds: int, retry_interval: float) -> None:
    """Espera solo la disponibilidad del backend, Qdrant y Memgraph."""
    base_url = api.rstrip("/")
    health_endpoint = f"{base_url}/api/v1/health"
    deadline = time.monotonic() + wait_seconds
    last_error = "sin respuesta"

    print(
        f"[seed] Esperando backend/Qdrant/Memgraph hasta {wait_seconds}s...",
        flush=True,
    )

    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        while time.monotonic() < deadline:
            try:
                health = client.get(health_endpoint)
                health.raise_for_status()
                health_payload = health.json()
                if health_payload.get("qdrant") and health_payload.get("memgraph"):
                    print("[seed] Backend, Qdrant y Memgraph disponibles.", flush=True)
                    return
                last_error = f"infraestructura todavía no disponible: {health_payload}"
            except (httpx.HTTPError, ValueError) as exc:
                last_error = str(exc)

            time.sleep(retry_interval)

    raise RuntimeError(
        "No se pudo disponer de backend/Qdrant/Memgraph dentro de "
        f"{wait_seconds}s. Último error: {last_error}"
    )


def _execute_seed(api: str, processing_timeout_seconds: float | None) -> list[dict[str, object]]:
    """Ejecuta la indexación sin reutilizar el timeout de readiness."""
    base_url = api.rstrip("/")
    seed_endpoint = f"{base_url}/api/v1/datasets/seed"

    read_timeout = (
        None
        if processing_timeout_seconds is None or processing_timeout_seconds <= 0
        else processing_timeout_seconds
    )
    timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=60.0, pool=60.0)
    timeout_label = "sin límite de lectura" if read_timeout is None else f"{read_timeout:g}s"

    print(
        "[seed] Iniciando precarga e indexación completa "
        f"(timeout de procesamiento: {timeout_label})...",
        flush=True,
    )
    started = time.monotonic()

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(seed_endpoint)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Falló la comunicación durante la precarga: {exc}") from exc

    elapsed = time.monotonic() - started
    if response.is_error:
        raise RuntimeError(
            "La precarga falló con un error no reintentable: "
            f"HTTP {response.status_code}: {response.text[:2000]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("La API devolvió una respuesta no JSON durante la precarga") from exc

    if not isinstance(payload, list):
        raise RuntimeError("La API devolvió un payload inesperado al cargar el dataset")

    print(f"[seed] Precarga completada en {elapsed:.1f}s.", flush=True)
    return payload


def seed_with_retry(
    api: str,
    wait_seconds: int,
    retry_interval: float,
    processing_timeout_seconds: float | None = None,
) -> list[dict[str, object]]:
    """Espera infraestructura y luego ejecuta una única precarga de larga duración."""
    _wait_for_infrastructure(api, wait_seconds, retry_interval)
    return _execute_seed(api, processing_timeout_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precarga el dataset de payment processing usando la API")
    parser.add_argument("--api", default="http://localhost:8000", help="URL base de la API")
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="Tiempo máximo para esperar a que backend, Qdrant y Memgraph estén disponibles",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=3.0,
        help="Segundos entre reintentos mientras se espera la infraestructura",
    )
    parser.add_argument(
        "--processing-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Timeout de lectura para la indexación completa; 0 significa sin límite. "
            "No afecta el timeout de disponibilidad de --wait-seconds."
        ),
    )
    args = parser.parse_args()

    wait_seconds = max(args.wait_seconds, 1)
    payload = seed_with_retry(
        args.api,
        wait_seconds,
        max(args.retry_interval, 0.2),
        args.processing_timeout_seconds,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
