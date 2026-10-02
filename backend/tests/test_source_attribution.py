from pathlib import Path

from pe.axiz.payment_knowledge.retrieval.source_attribution import (
    canonical_document_id,
    canonical_source_name,
    infer_source_from_text,
    parse_lightrag_context,
)


def test_canonical_source_and_document_id() -> None:
    name = "ef78b3d5f8f716b296b0f1f3__reconciliation.md"
    assert canonical_source_name(name) == "reconciliation.md"
    assert canonical_document_id(name) == "ef78b3d5f8f716b296b0f1f3"


def test_infer_source_from_text_usa_markdown_canonico(tmp_path: Path) -> None:
    (tmp_path / "aaa__authorization_codes.md").write_text(
        "# Código 05\nEl response code 05 significa Do not honor y operaciones revisa al emisor.",
        encoding="utf-8",
    )
    (tmp_path / "bbb__pci_tokenization.md").write_text(
        "# PCI DSS\nLa tokenización reemplaza el PAN por un token.",
        encoding="utf-8",
    )

    attribution = infer_source_from_text(
        "La tokenización reemplaza el PAN por un token.",
        tmp_path,
    )

    assert attribution is not None
    assert attribution.source == "pci_tokenization.md"
    assert attribution.document_id == "bbb"


def test_parse_lightrag_context_expone_fuentes_reales_en_orden(tmp_path: Path) -> None:
    text = r'''
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):
```json
{"reference_id": "1", "content": "# Conciliación\nEl adquirente genera clearing para conciliación."}
{"reference_id": "2", "content": "# Autorización\nEl código 05 significa Do not honor."}
```

Reference Document List (Each entry starts with a [reference_id]):
```
[1] ef78b3__reconciliation.md
[2] 83b32f__authorization_codes.md
```
'''
    items = parse_lightrag_context(text, tmp_path, top_k=5)

    assert [item["source"] for item in items] == [
        "reconciliation.md",
        "authorization_codes.md",
    ]
    assert [item["document_id"] for item in items] == ["ef78b3", "83b32f"]
    assert all(item["metadata"]["source_attribution"] == "reference_document_list" for item in items)
