import pytest

from voice_summary_bot.config import ConfigurationError, Settings


def test_bot_fails_closed_without_allowlist(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:example")
    monkeypatch.setenv("ALLOWED_USER_IDS", "")
    with pytest.raises(ConfigurationError, match="ALLOWED_USER_IDS"):
        Settings.load(tmp_path / "missing.env", require_bot=True)


def test_token_is_not_in_settings_repr():
    assert "private-value" not in repr(Settings(token="private-value"))


@pytest.mark.parametrize(
    "name,value",
    [
        ("MAX_DURATION_SECONDS", "601"),
        ("MAX_FILE_BYTES", "20000001"),
        ("SUMMARY_LANGUAGE", "invalid"),
        ("ALLOWED_USER_IDS", "-123"),
    ],
)
def test_invalid_configuration_is_rejected(monkeypatch, tmp_path, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError):
        Settings.load(tmp_path / "missing.env")


@pytest.mark.parametrize("duration", [300, 600])
def test_five_and_ten_minute_deployment_limits(monkeypatch, tmp_path, duration):
    monkeypatch.setenv("MAX_DURATION_SECONDS", str(duration))
    assert Settings.load(tmp_path / "missing.env").max_duration == duration


def test_default_duration_is_ten_minutes(monkeypatch, tmp_path):
    monkeypatch.delenv("MAX_DURATION_SECONDS", raising=False)
    assert Settings.load(tmp_path / "missing.env").max_duration == 600
