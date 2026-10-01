from pathlib import Path
from types import SimpleNamespace

from pe.axiz.raglight_service.main import RagLightEngine, configure_huggingface_environment


class FakeClient:
    def __init__(self, counts: list[int]) -> None:
        self.counts = iter(counts)

    def count(self, collection_name: str, exact: bool = True) -> SimpleNamespace:
        return SimpleNamespace(count=next(self.counts))


class FakeStore:
    def __init__(self, counts: list[int]) -> None:
        self.client = FakeClient(counts)
        self.ingested: list[str] = []

    def ingest(self, data_path: str) -> None:
        self.ingested.append(data_path)


def test_source_name_recupera_nombre_original() -> None:
    source = "/tmp/123456__authorization_codes.md"
    assert RagLightEngine._source_name(source) == "authorization_codes.md"


def test_validate_directory_rechaza_archivo(tmp_path: Path) -> None:
    engine = RagLightEngine()
    engine.canonical_dir = (tmp_path / "canonical").resolve()
    engine.canonical_dir.mkdir(parents=True)
    file_path = engine.canonical_dir / "documento.md"
    file_path.write_text("contenido", encoding="utf-8")

    try:
        engine._validate_directory(file_path)
    except ValueError as exc:
        assert "directorio" in str(exc)
    else:
        raise AssertionError("Debió rechazar un archivo individual")


def test_validate_directory_rechaza_path_fuera_del_canonico(tmp_path: Path) -> None:
    engine = RagLightEngine()
    engine.canonical_dir = (tmp_path / "canonical").resolve()
    engine.canonical_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "documento.md").write_text("contenido", encoding="utf-8")

    try:
        engine._validate_directory(outside)
    except ValueError as exc:
        assert "debe estar dentro" in str(exc)
    else:
        raise AssertionError("Debió rechazar un directorio fuera del canónico")


def test_index_exige_incremento_de_puntos(tmp_path: Path) -> None:
    engine = RagLightEngine()
    engine.canonical_dir = (tmp_path / "canonical").resolve()
    batch = engine.canonical_dir / "batch"
    batch.mkdir(parents=True)
    (batch / "a.md").write_text("contenido A", encoding="utf-8")
    (batch / "b.md").write_text("contenido B", encoding="utf-8")
    fake = FakeStore([10, 13])
    engine._vector_store = fake

    result = engine.index(str(batch))

    assert result.indexed is True
    assert result.files == 2
    assert result.points_before == 10
    assert result.points_after == 13
    assert fake.ingested == [str(batch.resolve())]


def test_index_falla_si_raglight_no_agrega_puntos(tmp_path: Path) -> None:
    engine = RagLightEngine()
    engine.canonical_dir = (tmp_path / "canonical").resolve()
    batch = engine.canonical_dir / "batch"
    batch.mkdir(parents=True)
    (batch / "a.md").write_text("contenido A", encoding="utf-8")
    engine._vector_store = FakeStore([10, 10])

    try:
        engine.index(str(batch))
    except RuntimeError as exc:
        assert "puntos_agregados=0" in str(exc)
    else:
        raise AssertionError("Debió fallar cuando RAGLight no agregó puntos")


def test_configure_huggingface_environment_propaga_aliases(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    assert configure_huggingface_environment() is True
    assert __import__("os").environ["HUGGING_FACE_HUB_TOKEN"] == "hf_test_token"
    assert __import__("os").environ["HUGGINGFACE_HUB_TOKEN"] == "hf_test_token"


def test_ready_inicializa_store_y_reporta_puntos(tmp_path: Path) -> None:
    engine = RagLightEngine()
    engine.canonical_dir = (tmp_path / "canonical").resolve()
    engine.canonical_dir.mkdir(parents=True)
    engine._vector_store = FakeStore([7])

    response = engine.ready()

    assert response.ready is True
    assert response.points == 7


def test_dependency_versions_reporta_paquetes(monkeypatch) -> None:
    import pe.axiz.raglight_service.main as module

    versions = {
        "raglight": "3.4.7",
        "langgraph": "1.0.5",
        "langgraph-prebuilt": "1.0.5",
        "langgraph-checkpoint": "3.0.1",
        "langgraph-sdk": "0.3.0",
        "qdrant-client": "1.17.0",
    }

    monkeypatch.setattr(module, "version", lambda package: versions[package])
    assert module.dependency_versions() == versions
