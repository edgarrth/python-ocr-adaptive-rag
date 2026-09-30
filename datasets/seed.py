from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Precarga el dataset de payment processing usando la API")
    parser.add_argument("--api", default="http://localhost:8000", help="URL base de la API")
    args = parser.parse_args()
    response = httpx.post(f"{args.api.rstrip('/')}/api/v1/datasets/seed", timeout=600)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
