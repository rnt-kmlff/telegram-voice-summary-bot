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
