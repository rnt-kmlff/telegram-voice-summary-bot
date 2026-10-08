"""Private Telegram adapter. Inference stays independent of the messaging layer."""

import asyncio
import contextlib
import fcntl
import io
import logging
import os
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from telegram import (
    BotCommand,
    BotCommandScopeChat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MenuButtonCommands,
    Update,
)
from telegram.constants import ChatType
from telegram.error import BadRequest, TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from .config import Settings
from .inference import (
    AudioDurationError,
    InferenceError,
    OllamaSummarizer,
    Summary,
    Transcript,
    WhisperTranscriber,
)
from .storage import Job, QueueFull, Store

LOGGER = logging.getLogger(__name__)
HELP = (
    "Forward an English or Russian voice message or audio file, up to {limit}. "
    "I'll return a summary and a Full transcript button.\n\n"
    "/language auto|en|ru — summary language\n"
    "/limit — choose a 5- or 10-minute audio limit\n"
    "/status — pending messages\n\n"
    "Audio is processed locally. Local transcripts expire after 24 hours by default; "
    "Telegram messages remain until deleted separately. "
    "Messages and replies still pass through Telegram. Check important details against the audio."
)


def duration_label(seconds: int) -> str:
    return f"{seconds // 60} minutes" if seconds % 60 == 0 else f"{seconds} seconds"


def limit_buttons(seconds: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"{'✓ ' if seconds == minutes * 60 else ''}{minutes} minutes",
                    callback_data=f"limit:{minutes}",
                )
                for minutes in (5, 10)
            ]
        ]
    )


class ProcessLock:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = data_dir / "bot.lock"
        self.file = None

    def __enter__(self):
        self.file = self.path.open("a+")
        self.path.chmod(0o600)
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError(
                "Another bot process is already using this data directory."
            ) from None
        return self

    def __exit__(self, *_):
        if self.file:
            self.file.close()


def cleanup_audio(data_dir: Path) -> Path:
    """Recover our own abandoned temporary directories while holding the process lock."""
    root = data_dir / "audio-tmp"
    if root.is_symlink():
        raise RuntimeError("The audio temporary directory must not be a symbolic link.")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    for child in root.iterdir():
        if (
            re.fullmatch(r"voice-(?:summary-)?[a-z0-9_]{8}", child.name)
            and child.is_dir()
            and not child.is_symlink()
        ):
            shutil.rmtree(child)
    return root


def authorized(update: Update, settings: Settings) -> bool:
    return bool(
        update.effective_user
        and update.effective_chat
        and update.effective_chat.type == ChatType.PRIVATE
        and update.effective_user.id in settings.allowed_user_ids
    )


def transcript_button(job_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Full transcript", callback_data=f"transcript:{job_id}")]]
    )


def render_summary(summary: Summary, language: str = "en", *, preview: bool = True) -> str:
    russian = language == "ru"
    translated = getattr(summary, "translated", False)
    abstractive = getattr(summary, "mode", "extractive") == "abstractive"
    heading = (
        ("Машинный перевод резюме:" if russian else "Machine-translated summary:")
        if translated
        else (
            ("Кратко:" if russian else "Summary:")
            if abstractive
            else ("Главное из сообщения:" if russian else "Key excerpts from the recording:")
        )
    )

    def display(text: str) -> str:
        return text if translated or abstractive else f"“{text}”"

    lines = [heading, display(summary.overview)]
    if summary.key_points:
        lines.extend(["", *[f"• {display(point)}" for point in summary.key_points]])
    for title, items in (
        ("Действия" if russian else "Actions", summary.actions),
        ("Неясности" if russian else "Uncertainties", summary.uncertainties),
    ):
        if items:
            lines.extend(["", f"{title}:", *[f"• {display(item)}" for item in items]])
    text = "\n".join(lines)
    # Never cut a sentence, negation or uncertainty just to fit a preview.
    if preview and len(text.encode("utf-16-le")) // 2 > 3600:
        notice = (
            "Полное резюме и расшифровка доступны по кнопке Full transcript."
            if russian
            else "The complete summary and transcript are available via Full transcript."
        )
        if len(summary.overview.encode("utf-16-le")) // 2 > 2500:
            return f"{heading}\n\n{notice}"
        return f"{heading}\n{display(summary.overview)}\n\n{notice}"
    return text


class VoiceBot:
    def __init__(self, settings: Settings, store: Store, transcriber, summarizer):
        self.settings = settings
        self.store = store
        self.transcriber = transcriber
        self.summarizer = summarizer
        self.worker_task: asyncio.Task | None = None
        self.worker_failed = False

    async def start(self, application: Application) -> None:
        self.store.recover()
        self.store.purge()
        self.worker_task = asyncio.create_task(self.worker(application), name="voice-worker")
        self.worker_task.add_done_callback(self.worker_finished)
        await self.register_commands(application)

    async def register_commands(self, application: Application) -> None:
        commands = [
            BotCommand("start", "How to use this bot"),
            BotCommand("help", "Show instructions and current audio limit"),
            BotCommand("language", "Choose summary language: auto, en or ru"),
            BotCommand("limit", "Choose a 5- or 10-minute audio limit"),
            BotCommand("status", "Show queue and current audio limit"),
        ]
        for user_id in sorted(self.settings.allowed_user_ids):
            try:
                async with asyncio.timeout(15):
                    await application.bot.set_my_commands(
                        commands, scope=BotCommandScopeChat(chat_id=user_id)
                    )
                    await application.bot.set_chat_menu_button(
                        chat_id=user_id, menu_button=MenuButtonCommands()
                    )
            except Exception as exc:
                # Menu availability must not stop the processing worker.
                LOGGER.warning("Command menu could not be updated (%s).", type(exc).__name__)

    def worker_finished(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            self.worker_failed = True
            LOGGER.error(
                "Processing worker stopped (%s); restart required.", type(task.exception()).__name__
            )

    async def stop(self, application: Application) -> None:
        if self.worker_task:
            self.worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker_task

    async def shutdown(self, application: Application) -> None:
        await self.summarizer.close()

    async def help(self, update: Update, context) -> None:
        if authorized(update, self.settings) and update.effective_message:
            await update.effective_message.reply_text(
                HELP.format(limit=duration_label(self.duration_limit(update.effective_user.id)))
            )
            LOGGER.info("Instructions delivered to an authorized account.")

    def duration_limit(self, user_id: int) -> int:
        return min(
            self.store.get_duration_limit(user_id, default=self.settings.max_duration),
            self.settings.max_duration,
        )

    def limit_text(self, user_id: int) -> str:
        text = (
            f"Audio limit: {duration_label(self.duration_limit(user_id))}.\n"
            "Choose 5 or 10 minutes. Changes apply to future recordings; "
            "queued recordings keep their accepted limit."
        )
        if self.settings.max_duration < 600:
            text += f"\nThis bot's maximum is {duration_label(self.settings.max_duration)}."
        return text

    def change_limit(self, user_id: int, choice: str) -> str | None:
        if choice not in {"5", "10"}:
            return "Use /limit 5 or /limit 10, or /limit to show the buttons."
        seconds = int(choice) * 60
        if seconds > self.settings.max_duration:
            return (
                f"This bot allows at most {duration_label(self.settings.max_duration)}. "
                f"Your audio limit remains {duration_label(self.duration_limit(user_id))}."
            )
        self.store.set_duration_limit(user_id, seconds)
        return None

    async def limit(self, update: Update, context) -> None:
        if not authorized(update, self.settings) or not update.effective_message:
            return
        if context.args:
            error = self.change_limit(
                update.effective_user.id, context.args[0] if len(context.args) == 1 else ""
            )
            if error:
                await update.effective_message.reply_text(error)
                return
        await update.effective_message.reply_text(
            self.limit_text(update.effective_user.id),
            reply_markup=limit_buttons(self.duration_limit(update.effective_user.id)),
        )

    async def limit_callback(self, update: Update, context) -> None:
        query = update.callback_query
        if not query:
            return
        if not authorized(update, self.settings):
            await query.answer("Unavailable.", show_alert=True)
            return
        error = self.change_limit(
            update.effective_user.id, (query.data or "").removeprefix("limit:")
        )
        if error:
            await query.answer(error, show_alert=True)
            return
        await query.answer(
            f"Audio limit: {duration_label(self.duration_limit(update.effective_user.id))}."
        )
        try:
            await query.edit_message_text(
                self.limit_text(update.effective_user.id),
                reply_markup=limit_buttons(self.duration_limit(update.effective_user.id)),
            )
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise

    async def language(self, update: Update, context) -> None:
        if not authorized(update, self.settings) or not update.effective_message:
            return
        if len(context.args) != 1 or context.args[0] not in {"auto", "en", "ru"}:
            await update.effective_message.reply_text(
                "Use /language auto, /language en or /language ru."
            )
            return
        self.store.set_language(update.effective_user.id, context.args[0])
        await update.effective_message.reply_text(f"Summary language: {context.args[0]}")

    async def status(self, update: Update, context) -> None:
        if authorized(update, self.settings) and update.effective_message:
            limit = f"Audio limit: {duration_label(self.duration_limit(update.effective_user.id))}."
            if self.worker_failed:
                await update.effective_message.reply_text(
                    f"Processing has stopped. Please restart the local bot service.\n{limit}"
                )
                return
            await update.effective_message.reply_text(
                f"Pending messages: {self.store.pending_count()}\n{limit}"
            )

    async def receive(self, update: Update, context) -> None:
        if not authorized(update, self.settings) or not update.effective_message:
            return
        message = update.effective_message
        if self.worker_failed:
            await message.reply_text(
                "Processing has stopped. Restart the local bot, then forward this recording again."
            )
            return
        media = message.voice or message.audio
        if not media:
            return
        duration = media.duration
        seconds = duration.total_seconds() if hasattr(duration, "total_seconds") else duration
        duration_limit = self.duration_limit(update.effective_user.id)
        if seconds is not None and seconds > duration_limit:
            hint = (
                " Use /limit to choose 10 minutes."
                if duration_limit < 600 and self.settings.max_duration >= 600
                else " Use /limit to view your setting."
            )
            await message.reply_text(
                f"Your audio limit is {duration_label(duration_limit)}. "
                f"Please send a recording within that limit.{hint}"
            )
            return
        if media.file_size and media.file_size > self.settings.max_file_bytes:
            await message.reply_text("Please send an audio file smaller than 20 MB.")
            return
        status = await message.reply_text(
            "Queued. I'll update this message when the summary is ready."
        )
        self.store.purge()
        try:
            job, created = self.store.enqueue(
                user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                message_id=message.message_id,
                file_id=media.file_id,
                file_unique_id=media.file_unique_id,
                status_message_id=status.message_id,
                max_pending=self.settings.max_pending_jobs,
                duration_limit=duration_limit,
            )
        except QueueFull:
            await status.edit_text("The queue is full. Please forward this recording again later.")
            return
        if not created and job.status_message_id != status.message_id:
            if job.state == "done" and job.summary_json:
                await status.edit_text(
                    render_summary(Summary.model_validate_json(job.summary_json), job.language),
                    reply_markup=transcript_button(job.id),
                )
            else:
                await status.edit_text("This recording is already queued or being processed.")

    async def transcript(self, update: Update, context) -> None:
        query = update.callback_query
        if not query:
            return
        if not authorized(update, self.settings):
            await query.answer("Unavailable.", show_alert=True)
            return
        self.store.purge()
        job_id = (query.data or "").removeprefix("transcript:")
        job = self.store.get(job_id, update.effective_user.id, update.effective_chat.id)
        if not job or not job.transcript:
            await query.answer("This transcript has expired or is unavailable.", show_alert=True)
            return
        await query.answer()
        text = job.transcript
        if job.summary_json:
            summary = Summary.model_validate_json(job.summary_json)
            text = (
                render_summary(summary, job.language, preview=False)
                + "\n\n--- Full transcript ---\n\n"
                + text
            )
        if job.removed_endings_json:
            text += (
                "\n\n--- Transcription cleanup ---\n"
                "Known subtitle-credit ending(s) were removed automatically. "
                "The original text and removal record are retained locally under the same "
                "expiry policy. Check the audio if that ending was intentional."
            )
        document = io.BytesIO(text.encode("utf-8"))
        await context.bot.send_document(
            chat_id=job.chat_id,
            document=document,
            filename="transcript.txt",
            caption="Automatic transcript. Check important details against the original audio.",
        )

    async def edit_status(self, application, job: Job, text: str, *, transcript=False) -> None:
        try:
            await application.bot.edit_message_text(
                chat_id=job.chat_id,
                message_id=job.status_message_id,
                text=text,
                reply_markup=transcript_button(job.id) if transcript else None,
            )
        except BadRequest as exc:
            # Replaying a recovered job may legitimately repeat the same edit.
            if "message is not modified" not in str(exc).lower():
                raise

    async def process_job(self, application, job: Job) -> None:
        # Authorization is checked again in case the allowlist changed after a restart.
        if job.user_id not in self.settings.allowed_user_ids or job.chat_id != job.user_id:
            self.store.fail(job.id)
            return
        await self.edit_status(application, job, "Transcribing your recording…")
        audio_root = self.settings.data_dir / "audio-tmp"
        audio_root.mkdir(mode=0o700, exist_ok=True)
        with TemporaryDirectory(prefix="voice-", dir=audio_root) as temporary:
            audio_path = Path(temporary) / "input.audio"
            audio_file = await application.bot.get_file(job.file_id)
            if audio_file.file_size and audio_file.file_size > self.settings.max_file_bytes:
                raise InferenceError("The file exceeds the download limit.")
            await audio_file.download_to_drive(audio_path)
            if audio_path.stat().st_size > self.settings.max_file_bytes:
                raise InferenceError("The downloaded file exceeds the limit.")
            transcript = await self.transcriber.transcribe(
                audio_path,
                language=self.settings.whisper_language,
                max_duration=min(job.duration_limit, self.settings.max_duration),
            )
        # Save the transcript before summarizing, so model failure does not lose it.
        self.store.save_transcript(
            job.id,
            transcript.text,
            transcript.language,
            raw_transcript=transcript.raw_text,
            removed_endings=transcript.removed_endings,
        )
        if not transcript.text.strip():
            self.store.fail(job.id)
            await self.edit_status(
                application,
                job,
                "No usable speech remained after transcription cleanup. "
                "Please send a clearer recording.",
            )
            return
        await self.edit_status(application, job, "Preparing your summary…", transcript=True)
        language = self.store.get_language(job.user_id, self.settings.summary_language)
        try:
            # The local audit must never become model input or part of the summary.
            summary = await self.summarizer.summarize(
                Transcript(transcript.text, transcript.language), output_language=language
            )
        except InferenceError:
            self.store.fail(job.id)
            await self.edit_status(
                application,
                job,
                "Your transcript is ready, but the summary could not be generated. "
                "Use Full transcript, or forward the recording again to retry.",
                transcript=True,
            )
            return
        output_language = transcript.language if language == "auto" else language
        self.store.complete(job.id, transcript.text, output_language, summary.model_dump_json())
        await self.edit_status(
            application, job, render_summary(summary, output_language), transcript=True
        )
        LOGGER.info("Summary delivered to an authorized account.")

    async def worker(self, application: Application) -> None:
        while True:
            job = self.store.next_job()
            if job is None:
                await asyncio.sleep(2)
                continue
            try:
                await self.process_job(application, job)
            except asyncio.CancelledError:
                raise
            except AudioDurationError:
                self.store.fail(job.id)
                accepted_limit = duration_label(min(job.duration_limit, self.settings.max_duration))
                with contextlib.suppress(TelegramError):
                    await self.edit_status(
                        application,
                        job,
                        f"This recording exceeds its accepted audio limit of {accepted_limit}. "
                        "Use /limit to check your setting, "
                        "then forward a recording within that limit.",
                    )
            except Exception as exc:
                self.store.fail(job.id)
                # Exception messages can contain URLs, tokens or source text; log only the type.
                LOGGER.warning("Recording processing failed (%s)", type(exc).__name__)
                with contextlib.suppress(TelegramError):
                    await self.edit_status(
                        application,
                        job,
                        "I couldn't finish this recording. Check the local services, "
                        "then forward it again to retry.",
                    )

    async def error(self, update, context) -> None:
        LOGGER.warning("Telegram handler failed (%s)", type(context.error).__name__)


def run_bot(settings: Settings) -> None:
    settings.validate_bot()
    os.umask(0o077)
    with ProcessLock(settings.data_dir):
        temporary_root = cleanup_audio(settings.data_dir)
        store = Store(settings.data_dir, settings.ttl_hours)
        try:
            transcriber = WhisperTranscriber(
                settings.model_path,
                binary=settings.whisper_binary,
                ffmpeg_binary=settings.ffmpeg_binary,
                max_duration=settings.max_duration,
                temporary_root=temporary_root,
            )
            summarizer = OllamaSummarizer(settings.ollama_base_url, settings.ollama_model)
            handlers = VoiceBot(settings, store, transcriber, summarizer)
            application = (
                Application.builder()
                .token(settings.token)
                .post_init(handlers.start)
                .post_stop(handlers.stop)
                .post_shutdown(handlers.shutdown)
                .build()
            )
            application.add_handler(CommandHandler(["start", "help"], handlers.help))
            application.add_handler(CommandHandler("language", handlers.language))
            application.add_handler(CommandHandler("limit", handlers.limit))
            application.add_handler(CommandHandler("status", handlers.status))
            application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handlers.receive))
            application.add_handler(
                CallbackQueryHandler(handlers.transcript, pattern=r"^transcript:")
            )
            application.add_handler(
                CallbackQueryHandler(handlers.limit_callback, pattern=r"^limit:")
            )
            application.add_error_handler(handlers.error)
            LOGGER.info("Private voice bot starting.")
            application.run_polling(
                allowed_updates=["message", "callback_query"],
                drop_pending_updates=False,
                bootstrap_retries=0,
            )
        finally:
            store.close()
