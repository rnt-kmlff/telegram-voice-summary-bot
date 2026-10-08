"""Command-line entry points, including a token-free local summary helper."""

import argparse
import asyncio
import copy
import json
import logging
import secrets
import shutil
import sys
from pathlib import Path
from typing import TextIO

import httpx
from telegram import Bot
from telegram.constants import ChatType
from telegram.error import TelegramError

from .config import ConfigurationError, Settings
from .inference import (
    MAX_TRANSCRIPT_CHARACTERS,
    InferenceError,
    OllamaSummarizer,
    Transcript,
    WhisperTranscriber,
)


class _PrivateLogFilter(logging.Filter):
    """Keep library exceptions useful without emitting credentials or source text."""

    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        if record.name == "voice_summary_bot" or record.name.startswith("voice_summary_bot."):
            return record
        if record.levelno < logging.WARNING:
            return False
        safe = copy.copy(record)
        exception_type = None
        if record.exc_info and record.exc_info[1] is not None:
            name = type(record.exc_info[1]).__name__
            exception_type = name if name.isidentifier() and len(name) <= 80 else "Exception"
        safe.name = "third_party"
        safe.msg = "Library event" + (f" ({exception_type})" if exception_type else "")
        safe.args = ()
        safe.exc_info = None
        safe.exc_text = None
        safe.stack_info = None
        # Python 3.12 handler filters can replace the record, so other handlers
        # never receive a record that this filter has modified in place.
        return safe


def configure_logging(*, stream: TextIO | None = None) -> None:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_PrivateLogFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


async def local_command(args, settings: Settings) -> None:
    if args.command == "summarize":
        if args.file.stat().st_size > MAX_TRANSCRIPT_CHARACTERS * 4:
            raise InferenceError("The transcript file exceeds the local helper's input limit.")
        transcript = Transcript(args.file.read_text(encoding="utf-8"), "auto")
    else:
        transcriber = WhisperTranscriber(
            settings.model_path,
            binary=settings.whisper_binary,
            ffmpeg_binary=settings.ffmpeg_binary,
            max_duration=settings.max_duration,
        )
        transcript = await transcriber.transcribe(args.file, language=settings.whisper_language)
    if not transcript.text.strip():
        raise InferenceError("No usable speech remained after transcription cleanup.")
    if args.command == "transcribe":
        print(transcript.text)
        return
    summarizer = OllamaSummarizer(settings.ollama_base_url, settings.ollama_model)
    try:
        summary = await summarizer.summarize(transcript, output_language=args.language)
    finally:
        await summarizer.close()
    result = {"summary": summary.model_dump()}
    if args.command == "process":
        result.update(transcript=transcript.text, source_language=transcript.language)
        if transcript.removed_endings:
            result["transcription_cleanup"] = {
                "raw_transcript": transcript.raw_text,
                "removed_endings": transcript.removed_endings,
            }
    print(json.dumps(result, ensure_ascii=False, indent=2))


async def doctor(settings: Settings) -> bool:
    checks = {
        "Whisper command": bool(shutil.which(settings.whisper_binary)),
        "FFmpeg command": bool(shutil.which(settings.ffmpeg_binary)),
        "Whisper model": settings.model_path.is_file(),
    }
    summarizer = OllamaSummarizer(settings.ollama_base_url, settings.ollama_model)
    try:
        # A metadata-only readiness check; never send a transcript to a missing model.
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            names = {entry["name"] for entry in response.json().get("models", [])}
            checks["Local Ollama service"] = True
            checks[f"Model {settings.ollama_model}"] = settings.ollama_model in names
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        checks["Local Ollama service and model"] = False
    finally:
        await summarizer.close()
    for name, ready in checks.items():
        print(f"{'OK' if ready else 'MISSING'}  {name}")
    try:
        settings.validate_bot()
    except ConfigurationError:
        print("SETUP NEEDED  Telegram token and/or allowed user IDs in .env")
    else:
        print("CONFIGURED  Telegram credentials (not contacted or verified)")
    return all(checks.values())


async def identify(settings: Settings) -> None:
    if not settings.token or ":" not in settings.token:
        raise ConfigurationError("First set TELEGRAM_BOT_TOKEN in your private .env file.")
    challenge = secrets.token_urlsafe(16)
    async with Bot(settings.token) as bot:
        identity = await bot.get_me()
        print("Setup only: use your NEW dedicated bot; keep the main bot process stopped.")
        print(f"Open https://t.me/{identity.username}?start={challenge} and press Start.")
        print("Waiting up to five minutes. Only the exact setup link will match.", flush=True)
        offset = None
        deadline = asyncio.get_running_loop().time() + 300
        while asyncio.get_running_loop().time() < deadline:
            updates = await bot.get_updates(offset=offset, timeout=10, allowed_updates=["message"])
            for update in updates:
                offset = update.update_id + 1
                message = update.effective_message
                if (
                    message
                    and message.chat.type == ChatType.PRIVATE
                    and message.from_user
                    and message.text == f"/start {challenge}"
                ):
                    print(f"Add to .env: ALLOWED_USER_IDS={message.from_user.id}")
                    return
        raise ConfigurationError("Setup timed out. Run identify again when ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Private Telegram voice summaries with local AI.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bot", help="Run the allowlisted Telegram bot.")
    subparsers.add_parser("doctor", help="Check local dependencies without contacting Telegram.")
    subparsers.add_parser("identify", help="Identify your account through a NEW dedicated bot.")
    for name in ("summarize", "transcribe", "process"):
        command = subparsers.add_parser(name)
        command.add_argument("file", type=Path)
        if name != "transcribe":
            command.add_argument("--language", choices=["auto", "en", "ru"], default="auto")
    args = parser.parse_args()
    configure_logging()
    try:
        settings = Settings.load(args.env_file, require_bot=args.command == "bot")
        if args.command == "bot":
            from .bot import run_bot

            run_bot(settings)
        elif args.command == "doctor":
            if not asyncio.run(doctor(settings)):
                raise SystemExit(1)
        elif args.command == "identify":
            asyncio.run(identify(settings))
        else:
            asyncio.run(local_command(args, settings))
    except (ConfigurationError, InferenceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except (OSError, UnicodeError, ValueError, RuntimeError, TelegramError) as exc:
        print(
            f"Operation failed ({type(exc).__name__}). Check the local configuration.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
