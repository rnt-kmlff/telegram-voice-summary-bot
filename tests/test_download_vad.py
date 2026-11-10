"""Pinned downloads are verified before publication; partial files are removed."""

import hashlib
import importlib.util
from pathlib import Path

import httpx
import pytest

from voice_summary_bot import speech_presence
from voice_summary_bot.errors import SpeechCheckUnavailableError

SPEC = importlib.util.spec_from_file_location(
    "download_vad", Path(__file__).resolve().parents[1] / "scripts/download_vad.py"
)
download_vad = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_vad)


@pytest.mark.parametrize(
    "mode", ["success", "wrong_hash", "too_large", "interrupted", "http_error"]
)
def test_pinned_stream_is_published_only_after_verification(tmp_path, monkeypatch, mode):
    approved = b"synthetic pinned model"
    monkeypatch.setattr(download_vad, "VAD_MODEL_BYTES", len(approved))
    monkeypatch.setattr(speech_presence, "VAD_MODEL_BYTES", len(approved))
    monkeypatch.setattr(speech_presence, "VAD_MODEL_SHA256", hashlib.sha256(approved).hexdigest())
    client_class = httpx.Client
    calls = []

    def serve(request):
        calls.append(str(request.url))
        if mode == "interrupted":
            raise httpx.ReadError("private diagnostic detail")
        body = {"wrong_hash": b"x" * len(approved), "too_large": approved + b"extra"}.get(
            mode, approved
        )
        return httpx.Response(503 if mode == "http_error" else 200, content=body)

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        return client_class(**kwargs, transport=httpx.MockTransport(serve))

    monkeypatch.setattr(download_vad.httpx, "Client", client)
    if mode == "success":
        path = download_vad.download(tmp_path)
        assert path.read_bytes() == approved
        # Repeated setup verifies the existing model without another download.
        assert download_vad.download(tmp_path) == path
    else:
        with pytest.raises((ValueError, httpx.HTTPError, SpeechCheckUnavailableError)):
            download_vad.download(tmp_path)
        assert not (tmp_path / download_vad.VAD_MODEL_FILENAME).exists()
    assert calls == [download_vad.VAD_MODEL_URL]
    assert speech_presence.VAD_MODEL_REVISION in calls[0]
    assert not list(tmp_path.glob(".vad-*"))


def test_existing_unverified_model_is_preserved_and_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / download_vad.VAD_MODEL_FILENAME
    path.write_bytes(b"unreviewed existing file")
    monkeypatch.setattr(download_vad.httpx, "Client", lambda **kwargs: pytest.fail("No network"))
    with pytest.raises(SpeechCheckUnavailableError):
        download_vad.download(tmp_path)
    assert path.read_bytes() == b"unreviewed existing file"
