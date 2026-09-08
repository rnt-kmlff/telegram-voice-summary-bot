import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest, NetworkError

from voice_summary_bot.bot import ProcessLock, VoiceBot, authorized, cleanup_audio, render_summary
from voice_summary_bot.config import Settings
from voice_summary_bot.inference import AudioDurationError, InferenceError, Summary, Transcript
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
    settings = Settings(allowed_user_ids=frozenset({1}), data_dir=tmp_path, max_duration=600)
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


@pytest.mark.parametrize("duration,size", [(601, 100), (10, 20_000_001)])
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


@pytest.mark.parametrize("language,heading", [("en", "Summary:"), ("ru", "Кратко:")])
def test_abstractive_json_renders_as_unquoted_summary(language, heading):
    summary = Summary.model_validate_json(
        json.dumps(
            {
                "overview": "Approval is pending.",
                "key_points": ["The proposed cost is $12,500."],
                "actions": ["The speaker will confirm the inspection date tomorrow."],
                "uncertainties": ["The inspection date is unknown."],
                "mode": "abstractive",
            }
        )
    )
    text = render_summary(summary, language)
    assert text.startswith(f"{heading}\nApproval is pending.\n")
    assert "• The proposed cost is $12,500." in text
    assert "• The speaker will confirm the inspection date tomorrow." in text
    assert "• The inspection date is unknown." in text
    assert "“" not in text and "”" not in text


def test_legacy_cached_json_keeps_excerpt_label_and_quotes():
    summary = Summary.model_validate_json(
        '{"overview":"I will confirm tomorrow.","key_points":[],"actions":[],"uncertainties":[]}'
    )
    assert summary.mode == "extractive"
    assert render_summary(summary) == (
        "Key excerpts from the recording:\n“I will confirm tomorrow.”"
    )


@pytest.mark.parametrize("mode", ["abstractive", "extractive"])
@pytest.mark.parametrize(
    "language,heading",
    [("en", "Machine-translated summary:"), ("ru", "Машинный перевод резюме:")],
)
def test_translated_json_remains_labeled_and_unquoted(mode, language, heading):
    summary = Summary.model_validate_json(
        json.dumps(
            {
                "overview": "Approval is pending.",
                "key_points": [],
                "actions": [],
                "uncertainties": [],
                "mode": mode,
                "translated": True,
            }
        )
    )
    assert render_summary(summary, language) == f"{heading}\nApproval is pending."


@pytest.mark.parametrize("mode", ["abstractive", "extractive"])
def test_long_summary_preview_keeps_complete_overview_and_download_keeps_all_text(mode):
    point = "A lengthy detail. " * 90
    summary = Summary(
        overview="Approval is pending; do not start work yet.",
        key_points=[point, point, point],
        actions=[],
        uncertainties=[],
        mode=mode,
    )
    preview = render_summary(summary)
    assert summary.overview in preview
    assert "Full transcript" in preview
    assert point not in preview
    assert len(preview.encode("utf-16-le")) // 2 <= 3600
    assert render_summary(summary, preview=False).count(point.strip()) == 3


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


async def test_limit_shows_current_selection_and_two_buttons(setup):
    _, _, adapter = setup
    event = update()
    await adapter.limit(event, SimpleNamespace(args=[]))
    reply = event.effective_message.reply_text.call_args
    assert "Audio limit: 10 minutes" in reply.args[0]
    assert "future recordings" in reply.args[0]
    assert "queued recordings keep their accepted limit" in reply.args[0]
    buttons = reply.kwargs["reply_markup"].inline_keyboard[0]
    assert [(button.text, button.callback_data) for button in buttons] == [
        ("5 minutes", "limit:5"),
        ("✓ 10 minutes", "limit:10"),
    ]


async def test_limit_command_persists_and_help_status_use_selection(setup):
    settings, _, adapter = setup
    event = update()
    await adapter.limit(event, SimpleNamespace(args=["5"]))
    reloaded = Store(settings.data_dir)
    try:
        assert reloaded.get_duration_limit(1) == 300
    finally:
        reloaded.close()
    await adapter.help(event, None)
    assert "up to 5 minutes" in event.effective_message.reply_text.call_args.args[0]
    await adapter.status(event, None)
    assert "Audio limit: 5 minutes" in event.effective_message.reply_text.call_args.args[0]


@pytest.mark.parametrize("args", [["15"], ["0"], ["5", "10"], ["600"]])
async def test_invalid_limit_command_does_not_change_preference(setup, args):
    _, store, adapter = setup
    store.set_duration_limit(1, 300)
    event = update()
    await adapter.limit(event, SimpleNamespace(args=args))
    assert store.get_duration_limit(1) == 300
    assert "Use /limit 5 or /limit 10" in event.effective_message.reply_text.call_args.args[0]


@pytest.mark.parametrize("user_id,chat_type", [(2, "private"), (1, "group")])
async def test_limit_command_and_callback_reject_unauthorized_context(setup, user_id, chat_type):
    _, store, adapter = setup
    event = update(user_id=user_id, chat_type=chat_type)
    event.callback_query = SimpleNamespace(
        data="limit:5", answer=AsyncMock(), edit_message_text=AsyncMock()
    )
    await adapter.limit(event, SimpleNamespace(args=["5"]))
    await adapter.limit_callback(event, None)
    event.effective_message.reply_text.assert_not_awaited()
    event.callback_query.edit_message_text.assert_not_awaited()
    event.callback_query.answer.assert_awaited_once_with("Unavailable.", show_alert=True)
    assert store.get_duration_limit(user_id) == 600


async def test_limit_callback_updates_selection_and_handles_repeated_click(setup):
    _, store, adapter = setup
    event = update()
    event.callback_query = SimpleNamespace(
        data="limit:5", answer=AsyncMock(), edit_message_text=AsyncMock()
    )
    await adapter.limit_callback(event, None)
    assert store.get_duration_limit(1) == 300
    assert "Audio limit: 5 minutes" in event.callback_query.edit_message_text.call_args.args[0]
    event.callback_query.edit_message_text.side_effect = BadRequest("Message is not modified")
    await adapter.limit_callback(event, None)
    assert store.get_duration_limit(1) == 300


async def test_forged_limit_callback_cannot_raise_limit(setup):
    _, store, adapter = setup
    event = update()
    event.callback_query = SimpleNamespace(
        data="limit:60", answer=AsyncMock(), edit_message_text=AsyncMock()
    )
    await adapter.limit_callback(event, None)
    assert store.get_duration_limit(1) == 600
    event.callback_query.edit_message_text.assert_not_awaited()
    assert event.callback_query.answer.call_args.kwargs["show_alert"] is True


async def test_limit_cannot_exceed_deployment_ceiling(tmp_path):
    settings = Settings(allowed_user_ids=frozenset({1}), data_dir=tmp_path, max_duration=300)
    store = Store(tmp_path)
    try:
        store.set_duration_limit(1, 600)
        adapter = VoiceBot(settings, store, None, None)
        event = update()
        await adapter.limit(event, SimpleNamespace(args=["10"]))
        assert "at most 5 minutes" in event.effective_message.reply_text.call_args.args[0]
        assert "remains 5 minutes" in event.effective_message.reply_text.call_args.args[0]
        assert adapter.duration_limit(1) == 300
        event.callback_query = SimpleNamespace(
            data="limit:10", answer=AsyncMock(), edit_message_text=AsyncMock()
        )
        await adapter.limit_callback(event, None)
        assert "at most 5 minutes" in event.callback_query.answer.call_args.args[0]
        event.callback_query.edit_message_text.assert_not_awaited()
    finally:
        store.close()


async def test_six_minute_audio_rejected_at_five_then_accepted_at_ten(setup):
    _, store, adapter = setup
    store.set_duration_limit(1, 300)
    media = SimpleNamespace(duration=360, file_size=400, file_id="file", file_unique_id="unique")
    event = update(voice=media)
    await adapter.receive(event, None)
    assert store.pending_count() == 0
    assert "audio limit is 5 minutes" in event.effective_message.reply_text.call_args.args[0]
    await adapter.limit(update(), SimpleNamespace(args=["10"]))
    await adapter.receive(event, None)
    accepted = store.next_job()
    assert accepted.duration_limit == 600


@pytest.mark.parametrize(
    "accepted,subsequent,deployment_cap", [(300, 600, 600), (600, 300, 600), (600, 600, 300)]
)
async def test_processing_uses_accepted_limit_even_after_preference_changes(
    setup, accepted, subsequent, deployment_cap
):
    _, store, adapter = setup
    store.set_duration_limit(1, accepted)
    # Telegram duration is untrusted; the decoder receives the snapshot even when metadata is short.
    media = SimpleNamespace(duration=1, file_size=400, file_id="file", file_unique_id="unique")
    await adapter.receive(update(voice=media), None)
    store.set_duration_limit(1, subsequent)
    adapter.settings = replace(adapter.settings, max_duration=deployment_cap)
    adapter.transcriber.max_duration = 600
    job = store.next_job()

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
    assert adapter.transcriber.transcribe.call_args.kwargs["max_duration"] == min(
        accepted, deployment_cap
    )
    assert adapter.transcriber.max_duration == 600


async def test_actual_duration_rejection_shows_accepted_limit(setup):
    _, store, adapter = setup
    job = enqueue(store)
    adapter.process_job = AsyncMock(side_effect=AudioDurationError("too long"))
    delivered = asyncio.Event()

    async def edited(**kwargs):
        delivered.set()

    app = SimpleNamespace(bot=SimpleNamespace(edit_message_text=AsyncMock(side_effect=edited)))
    task = asyncio.create_task(adapter.worker(app))
    try:
        await asyncio.wait_for(delivered.wait(), timeout=2)
        assert store.get(job.id, 1, 1).state == "failed"
        assert (
            "accepted audio limit of 5 minutes"
            in app.bot.edit_message_text.call_args.kwargs["text"]
        )
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_commands_menu_is_scoped_to_authorized_private_chat(setup):
    _, _, adapter = setup
    app = SimpleNamespace(
        bot=SimpleNamespace(set_my_commands=AsyncMock(), set_chat_menu_button=AsyncMock())
    )
    await adapter.register_commands(app)
    call = app.bot.set_my_commands.call_args
    assert [command.command for command in call.args[0]] == [
        "start",
        "help",
        "language",
        "limit",
        "status",
    ]
    assert call.kwargs["scope"].type == "chat"
    assert call.kwargs["scope"].chat_id == 1
    assert app.bot.set_chat_menu_button.call_args.kwargs["chat_id"] == 1


async def test_menu_failure_does_not_stop_worker_or_log_exception_content(setup, caplog):
    _, _, adapter = setup
    app = SimpleNamespace(
        bot=SimpleNamespace(
            set_my_commands=AsyncMock(side_effect=NetworkError("private request detail")),
            set_chat_menu_button=AsyncMock(),
        )
    )
    await adapter.start(app)
    try:
        await asyncio.sleep(0)
        assert not adapter.worker_task.done()
        assert not adapter.worker_failed
        assert "Command menu could not be updated" in caplog.text
        assert "private request detail" not in caplog.text
    finally:
        await adapter.stop(app)
