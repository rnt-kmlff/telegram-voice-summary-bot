from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voice_summary_bot.bot import ProcessLock, VoiceBot, authorized, cleanup_audio, render_summary
from voice_summary_bot.config import Settings
from voice_summary_bot.inference import InferenceError, Summary, Transcript
from voice_summary_bot.storage import Store


def update(user_id=1, chat_type="private", **media):
    message = SimpleNamespace(
        voice=media.get("voice"),
        audio=media.get("audio"),
        message_id=10,
        reply_text=AsyncMock(return_value=SimpleNamespace(message_id=20, edit_text=AsyncMock())),
    )
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=user_id, type=chat_type),
        effective_message=message,
    )


@pytest.fixture
def setup(tmp_path):
    settings = Settings(allowed_user_ids=frozenset({1}), data_dir=tmp_path)
    store = Store(tmp_path)
    transcriber = SimpleNamespace(transcribe=AsyncMock(return_value=Transcript("Hello", "en")))
    summarizer = SimpleNamespace(
        summarize=AsyncMock(
            return_value=Summary(overview="Hello", key_points=[], actions=[], uncertainties=[])
        )
    )
    adapter = VoiceBot(settings, store, transcriber, summarizer)
    yield settings, store, adapter
    store.close()


def test_private_owner_only(setup):
    settings, _, _ = setup
    assert authorized(update(), settings)
    assert not authorized(update(user_id=2), settings)
    assert not authorized(update(chat_type="group"), settings)


@pytest.mark.parametrize("duration,size", [(301, 100), (10, 20_000_001)])
async def test_reject_oversize_before_queue(setup, duration, size):
    _, store, adapter = setup
    event = update(voice=SimpleNamespace(duration=duration, file_size=size))
    await adapter.receive(event, None)
    assert store.pending_count() == 0
    assert event.effective_message.reply_text.await_count == 1


async def test_unauthorized_audio_does_not_acknowledge_or_download(setup):
    _, store, adapter = setup
    event = update(user_id=2, voice=object())
    await adapter.receive(event, None)
    event.effective_message.reply_text.assert_not_awaited()
    assert store.pending_count() == 0


async def test_forwarded_voice_is_queued_and_deduplicated(setup):
    _, store, adapter = setup
    media = SimpleNamespace(duration=30, file_size=400, file_id="file", file_unique_id="unique")
    await adapter.receive(update(voice=media), None)
    await adapter.receive(update(voice=media), None)
    assert store.pending_count() == 1


def enqueue(store):
    return store.enqueue(1, 1, 10, "file", "unique", 20)[0]


async def test_download_cleanup_and_summary_success(setup):
    settings, store, adapter = setup
    job = enqueue(store)
    store.next_job()

    async def download(destination):
        destination.write_bytes(b"synthetic fixture")

    telegram_file = SimpleNamespace(file_size=20, download_to_drive=AsyncMock(side_effect=download))
    app = SimpleNamespace(
        bot=SimpleNamespace(
            get_file=AsyncMock(return_value=telegram_file), edit_message_text=AsyncMock()
        )
    )
    await adapter.process_job(app, job)
    assert store.get(job.id, 1, 1).state == "done"
    assert store.get(job.id, 1, 1).transcript == "Hello"
    assert not list(settings.data_dir.glob("voice-*"))
    adapter.summarizer.summarize.assert_awaited_once()
    assert "Full transcript" in str(app.bot.edit_message_text.call_args)


async def test_summary_failure_preserves_transcript_and_allows_retry(setup):
    settings, store, adapter = setup
    job = enqueue(store)
    store.next_job()
    adapter.summarizer.summarize.side_effect = InferenceError("offline")

    async def download(destination):
        destination.write_bytes(b"synthetic fixture")

    app = SimpleNamespace(
        bot=SimpleNamespace(
            get_file=AsyncMock(
                return_value=SimpleNamespace(
                    file_size=20, download_to_drive=AsyncMock(side_effect=download)
                )
            ),
            edit_message_text=AsyncMock(),
        )
    )
    await adapter.process_job(app, job)
    stored = store.get(job.id, 1, 1)
    assert stored.state == "failed"
    assert stored.transcript == "Hello"
    assert not list(settings.data_dir.glob("voice-*"))
    assert "transcript is ready" in app.bot.edit_message_text.call_args.kwargs["text"]


async def test_callback_does_not_expose_another_owners_transcript(setup):
    _, store, adapter = setup
    job = enqueue(store)
    store.complete(job.id, "private", "en", "{}")
    event = update(user_id=2)
    event.callback_query = SimpleNamespace(data=f"transcript:{job.id}", answer=AsyncMock())
    context = SimpleNamespace(bot=SimpleNamespace(send_document=AsyncMock()))
    await adapter.transcript(event, context)
    context.bot.send_document.assert_not_awaited()


def test_summary_stays_within_telegram_utf16_limit():
    summary = Summary(
        overview="😀" * 400,
        key_points=["😀" * 400] * 3,
        actions=["😀" * 400] * 5,
        uncertainties=["😀" * 400] * 5,
    )
    assert len(render_summary(summary).encode("utf-16-le")) // 2 <= 3600


def test_second_poller_lock_is_rejected(tmp_path: Path):
    with (
        ProcessLock(tmp_path),
        pytest.raises(RuntimeError, match="Another bot"),
        ProcessLock(tmp_path),
    ):
        pass


async def test_worker_failure_rejects_new_inference(setup):
    _, store, adapter = setup
    adapter.worker_failed = True
    event = update(voice=object())
    await adapter.receive(event, None)
    assert store.pending_count() == 0
    assert "stopped" in event.effective_message.reply_text.call_args.args[0]


def test_crash_cleanup_only_removes_our_temporary_directories(tmp_path):
    root = cleanup_audio(tmp_path)
    ours = root / "voice-summary-abcd1234"
    ours.mkdir()
    (ours / "audio.wav").write_bytes(b"synthetic")
    preserved = root / "unrelated"
    preserved.mkdir()
    cleanup_audio(tmp_path)
    assert not ours.exists()
    assert preserved.exists()
