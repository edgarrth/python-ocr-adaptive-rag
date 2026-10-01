from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import httpx


def _load_seed_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "datasets" / "seed.py"
    spec = importlib.util.spec_from_file_location("axiz_dataset_seed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "ok"

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400

    def raise_for_status(self) -> None:
        if self.is_error:
            request = httpx.Request("GET", "http://test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> object:
        return self._payload


def test_processing_request_has_no_read_timeout_by_default(monkeypatch) -> None:
    seed = _load_seed_module()
    created_timeouts: list[httpx.Timeout] = []

    class FakeClient:
        def __init__(self, *, timeout: httpx.Timeout) -> None:
            created_timeouts.append(timeout)

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, _url: str) -> _Response:
            return _Response(200, {"qdrant": True, "memgraph": True})

        def post(self, _url: str) -> _Response:
            return _Response(200, [{"source": "authorization_codes.md"}])

    monkeypatch.setattr(seed.httpx, "Client", FakeClient)
    result = seed.seed_with_retry("http://backend:8000", 1, 0.2)

    assert result == [{"source": "authorization_codes.md"}]
    assert len(created_timeouts) == 2
    assert created_timeouts[0].read == 10.0
    assert created_timeouts[1].read is None


def test_processing_timeout_can_be_configured(monkeypatch) -> None:
    seed = _load_seed_module()
    created_timeouts: list[httpx.Timeout] = []

    class FakeClient:
        def __init__(self, *, timeout: httpx.Timeout) -> None:
            created_timeouts.append(timeout)

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, _url: str) -> _Response:
            return _Response(200, {"qdrant": True, "memgraph": True})

        def post(self, _url: str) -> _Response:
            return _Response(200, [])

    monkeypatch.setattr(seed.httpx, "Client", FakeClient)
    seed.seed_with_retry("http://backend:8000", 1, 0.2, processing_timeout_seconds=3600)

    assert created_timeouts[1].read == 3600.0
