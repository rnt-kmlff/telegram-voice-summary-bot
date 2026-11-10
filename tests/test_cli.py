"""Check emitted log output without reading configuration or contacting Telegram."""

import io
import logging
import sys

import pytest
from telegram.error import InvalidToken

from voice_summary_bot.cli import configure_logging


@pytest.fixture
def output():
    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    stream = io.StringIO()
    configure_logging(stream=stream)
    try:
        yield stream
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


def test_telegram_invalid_token_traceback_is_not_emitted(output):
    credential = "123456:FAKE_CREDENTIAL_FOR_TEST"
    try:
        raise InvalidToken(f"The token {credential} was rejected by the server")
    except InvalidToken:
        logging.getLogger("telegram.ext").exception(
            "Bootstrap failed for %s; source text was confidential",
            credential,
            stack_info=True,
        )
    emitted = output.getvalue()
    assert emitted == "ERROR third_party: Library event (InvalidToken)\n"
    assert credential not in emitted
    assert "confidential" not in emitted
    assert "Traceback" not in emitted and "Stack" not in emitted


def test_http_urls_and_informational_library_events_are_not_emitted(output):
    url = "https://api.telegram.org/bot123456:FAKE_TEST_ONLY/getUpdates"
    logger = logging.getLogger("httpx")
    logger.info("HTTP Request: %s", url)
    assert output.getvalue() == ""
    logger.warning("HTTP retry to %s with private response text", url)
    assert output.getvalue() == "WARNING third_party: Library event\n"
    assert url not in output.getvalue()


def test_explicitly_safe_application_warning_is_retained(output):
    logging.getLogger("voice_summary_bot.bot").warning(
        "Recording processing failed (%s)", "InferenceError"
    )
    assert output.getvalue() == (
        "WARNING voice_summary_bot.bot: Recording processing failed (InferenceError)\n"
    )


def test_filter_strips_cached_exception_text_without_mutating_shared_record(output):
    try:
        raise InvalidToken("synthetic-private-credential")
    except InvalidToken:
        exception_info = sys.exc_info()
    record = logging.LogRecord(
        "telegram.ext",
        logging.ERROR,
        __file__,
        1,
        "Private URL %s",
        ("https://example.invalid/private",),
        exception_info,
    )
    record.exc_text = "cached traceback containing synthetic-private-credential"
    record.stack_info = "private stack details"
    logging.getLogger().handlers[0].handle(record)
    assert output.getvalue() == "ERROR third_party: Library event (InvalidToken)\n"
    assert record.msg == "Private URL %s"
    assert record.args == ("https://example.invalid/private",)
    assert record.exc_info is exception_info
    assert record.exc_text == "cached traceback containing synthetic-private-credential"
    assert record.stack_info == "private stack details"


@pytest.mark.parametrize("command", ["process", "transcribe"])
async def test_audio_cli_returns_cleaned_text_and_process_audit(monkeypatch, capsys, command):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from voice_summary_bot import cli
    from voice_summary_bot.config import Settings
    from voice_summary_bot.inference import Summary, Transcript

    credit = "Субтитры сделал DimaTorzok"
    raw = "Please confirm. " + credit
    transcript = Transcript("Please confirm.", "en", raw, (credit,))
    monkeypatch.setattr(
        cli,
        "WhisperTranscriber",
        lambda *a, **kw: SimpleNamespace(transcribe=AsyncMock(return_value=transcript)),
    )
    summarizer = SimpleNamespace(
        summarize=AsyncMock(
            return_value=Summary(
                overview="Please confirm.", key_points=[], actions=[], uncertainties=[]
            )
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr(cli, "OllamaSummarizer", lambda *a: summarizer)
    await cli.local_command(
        SimpleNamespace(command=command, file=Path("unused"), language="auto"), Settings()
    )
    emitted = capsys.readouterr().out
    if command == "transcribe":
        assert emitted == "Please confirm.\n"
        summarizer.summarize.assert_not_awaited()
    else:
        payload = json.loads(emitted)
        assert payload["transcript"] == "Please confirm."
        assert payload["transcription_cleanup"] == {
            "raw_transcript": raw,
            "removed_endings": [credit],
        }
        summarizer.close.assert_awaited_once()


async def test_text_only_summary_does_not_filter_user_supplied_text(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from voice_summary_bot import cli
    from voice_summary_bot.config import Settings
    from voice_summary_bot.inference import Summary

    path = tmp_path / "source.txt"
    source = "Субтитры сделал DimaTorzok"
    path.write_text(source)
    summarizer = SimpleNamespace(
        summarize=AsyncMock(
            return_value=Summary(overview=source, key_points=[], actions=[], uncertainties=[])
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr(cli, "OllamaSummarizer", lambda *a: summarizer)
    await cli.local_command(
        SimpleNamespace(command="summarize", file=path, language="auto"), Settings()
    )
    assert summarizer.summarize.call_args.args[0].text == source
    assert "transcription_cleanup" not in capsys.readouterr().out


@pytest.mark.parametrize("missing", [None, "model", "command"])
async def test_doctor_requires_detector_command_and_verified_model(
    monkeypatch, tmp_path, capsys, missing
):
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import httpx

    from voice_summary_bot import cli
    from voice_summary_bot.config import Settings
    from voice_summary_bot.errors import SpeechCheckUnavailableError

    model = tmp_path / "whisper.bin"
    model.touch()
    settings = replace(Settings(), model_path=model)
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda name: None if missing == "command" and name == settings.vad_binary else name,
    )

    def verify(path):
        assert path == settings.vad_model_path
        if missing == "model":
            raise SpeechCheckUnavailableError()

    monkeypatch.setattr(cli, "verify_vad_model", verify)
    monkeypatch.setattr(cli, "OllamaSummarizer", lambda *a: SimpleNamespace(close=AsyncMock()))
    client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"models": [{"name": settings.ollama_model}]})
    )
    monkeypatch.setattr(cli.httpx, "AsyncClient", lambda **kw: client(**kw, transport=transport))
    assert await cli.doctor(settings) is (missing is None)
    output = capsys.readouterr().out
    if missing:
        assert "MISSING" in output
