from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_seed_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "datasets" / "seed.py"
    spec = importlib.util.spec_from_file_location("axiz_dataset_seed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PingAdapter:
    def __init__(self, values: list[bool], *, available: bool = True) -> None:
        self.values = list(values)
        self.available = available

    def ping(self) -> bool:
        if not self.values:
            return True
        return self.values.pop(0)


class _Result:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"source": "authorization_codes.md"}


def test_wait_for_infrastructure_does_not_depend_on_backend_http() -> None:
    seed = _load_seed_module()
    container = SimpleNamespace(
        qdrant=_PingAdapter([False, True]),
        memgraph=_PingAdapter([True, True]),
        raglight=_PingAdapter([True, True]),
    )

    asyncio.run(seed._wait_for_infrastructure(container, wait_seconds=2, retry_interval=0.01))


def test_seed_direct_uses_isolated_container_and_closes_it(monkeypatch) -> None:
    seed = _load_seed_module()
    closed = False

    class FakeIngestion:
        async def ingest_dataset(self) -> list[_Result]:
            return [_Result()]

    class FakeContainer:
        def __init__(self) -> None:
            self.qdrant = _PingAdapter([True])
            self.memgraph = _PingAdapter([True])
            self.raglight = _PingAdapter([True])
            self.ingestion = FakeIngestion()

        async def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(seed, "_create_container", FakeContainer)
    result = asyncio.run(seed.seed_direct(wait_seconds=1, retry_interval=0.01))

    assert result == [{"source": "authorization_codes.md"}]
    assert closed is True
