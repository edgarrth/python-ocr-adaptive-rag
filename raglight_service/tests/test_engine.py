from pathlib import Path

from pe.axiz.raglight_service.main import RagLightEngine


def test_source_name_recupera_nombre_original() -> None:
    source = "/tmp/123456__authorization_codes.md.md"
    assert RagLightEngine._source_name(source) == "authorization_codes.md"


def test_validate_path_rechaza_archivo_fuera_del_canonico(tmp_path: Path) -> None:
    engine = RagLightEngine()
    engine.canonical_dir = (tmp_path / "canonical").resolve()
    engine.canonical_dir.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("contenido", encoding="utf-8")

    try:
        engine._validate_path(outside)
    except ValueError as exc:
        assert "debe estar dentro" in str(exc)
    else:
        raise AssertionError("Debió rechazar un archivo fuera del directorio canónico")
