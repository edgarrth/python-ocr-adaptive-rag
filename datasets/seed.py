from __future__ import annotations

import argparse
import json
import time

import httpx


def seed_with_retry(api: str, wait_seconds: int, retry_interval: float) -> list[dict[str, object]]:
    """Espera al backend y carga el dataset cuando toda la ruta de ingesta está disponible."""
    endpoint = f"{api.rstrip('/')}/api/v1/datasets/seed"
    deadline = time.monotonic() + wait_seconds
    last_error = "sin respuesta"

    with httpx.Client(timeout=600) as client:
        while time.monotonic() < deadline:
            try:
                response = client.post(endpoint)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise RuntimeError("La API devolvió un payload inesperado al cargar el dataset")
                return payload
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                last_error = str(exc)
                time.sleep(retry_interval)

    raise RuntimeError(f"No se pudo precargar el dataset dentro de {wait_seconds}s. Último error: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precarga el dataset de payment processing usando la API")
    parser.add_argument("--api", default="http://localhost:8000", help="URL base de la API")
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="Tiempo máximo para reintentar mientras el backend o la infraestructura terminan de iniciar",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=3.0,
        help="Segundos entre reintentos durante el bootstrap",
    )
    args = parser.parse_args()

    wait_seconds = max(args.wait_seconds, 1)
    payload = seed_with_retry(args.api, wait_seconds, max(args.retry_interval, 0.2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
