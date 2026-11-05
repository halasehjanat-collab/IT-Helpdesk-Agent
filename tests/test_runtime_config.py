"""Project-local configuration lookup, isolated from machine credentials."""

from services import runtime_config


def test_blank_process_key_falls_back_to_project_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=fixture-key\n", encoding="utf-8")
    monkeypatch.setattr(runtime_config, "ENV_FILE", env_file)
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")

    assert runtime_config.openrouter_api_key() == "fixture-key"


def test_nonblank_process_key_takes_precedence(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=fixture-key\n", encoding="utf-8")
    monkeypatch.setattr(runtime_config, "ENV_FILE", env_file)
    monkeypatch.setenv("OPENROUTER_API_KEY", "process-key")

    assert runtime_config.openrouter_api_key() == "process-key"


def test_missing_key_stays_unconfigured(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_config, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert runtime_config.openrouter_api_key() is None
