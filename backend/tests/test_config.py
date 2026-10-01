from pe.axiz.payment_knowledge.config import Settings


def test_cors_origins_accepts_csv() -> None:
    settings = Settings(cors_origins="http://localhost:4200,http://localhost:8080")
    assert settings.parsed_cors_origins() == [
        "http://localhost:4200",
        "http://localhost:8080",
    ]


def test_cors_origins_accepts_json_array() -> None:
    settings = Settings(cors_origins='["http://localhost:4200","http://localhost:8080"]')
    assert settings.parsed_cors_origins() == [
        "http://localhost:4200",
        "http://localhost:8080",
    ]


def test_hf_token_se_lee_como_setting() -> None:
    settings = Settings(hf_token="hf_test")
    assert settings.hf_token == "hf_test"
