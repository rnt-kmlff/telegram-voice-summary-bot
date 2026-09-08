"""Configuration without network access or credential logging."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    pass


def positive_int(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        raise ConfigurationError(f"{name} must be a positive integer.") from None
    if not 1 <= value <= maximum:
        raise ConfigurationError(f"{name} must be between 1 and {maximum}.")
    return value


def language(name: str) -> str:
    value = os.getenv(name, "auto").lower().strip()
    if value not in {"auto", "en", "ru"}:
        raise ConfigurationError(f"{name} must be auto, en or ru.")
    return value


@dataclass(frozen=True)
class Settings:
    token: str = field(default="", repr=False)
    allowed_user_ids: frozenset[int] = frozenset()
    model_path: Path = Path("models/ggml-large-v3-turbo.bin")
    whisper_binary: str = "whisper-cli"
    ffmpeg_binary: str = "ffmpeg"
    whisper_language: str = "auto"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:9b"
    summary_language: str = "auto"
    data_dir: Path = Path("data")
    max_duration: int = 600
    max_file_bytes: int = 20_000_000
    max_pending_jobs: int = 10
    ttl_hours: int = 24

    @classmethod
    def load(cls, env_file: Path = Path(".env"), *, require_bot: bool = False) -> "Settings":
        load_dotenv(env_file, override=False)
        raw_ids = os.getenv("ALLOWED_USER_IDS", "")
        try:
            ids = frozenset(int(item.strip()) for item in raw_ids.split(",") if item.strip())
        except ValueError:
            raise ConfigurationError("ALLOWED_USER_IDS must contain numeric user IDs.") from None
        if any(uid <= 0 for uid in ids):
            raise ConfigurationError("ALLOWED_USER_IDS must contain positive user IDs.")
        settings = cls(
            token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            allowed_user_ids=ids,
            model_path=Path(os.getenv("WHISPER_MODEL_PATH", "models/ggml-large-v3-turbo.bin"))
            .expanduser()
            .resolve(),
            whisper_binary=os.getenv("WHISPER_BINARY", "whisper-cli"),
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg"),
            whisper_language=language("WHISPER_LANGUAGE"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
            summary_language=language("SUMMARY_LANGUAGE"),
            data_dir=Path(os.getenv("DATA_DIR", "data")).expanduser().resolve(),
            max_duration=positive_int("MAX_DURATION_SECONDS", 600, 600),
            max_file_bytes=positive_int("MAX_FILE_BYTES", 20_000_000, 20_000_000),
            max_pending_jobs=positive_int("MAX_PENDING_JOBS", 10, 100),
            ttl_hours=positive_int("TRANSCRIPT_TTL_HOURS", 24, 168),
        )
        if require_bot:
            settings.validate_bot()
        return settings

    def validate_bot(self) -> None:
        if not self.token or ":" not in self.token:
            raise ConfigurationError("Set TELEGRAM_BOT_TOKEN in your private .env file.")
        if not self.allowed_user_ids:
            raise ConfigurationError("Set ALLOWED_USER_IDS before starting the bot.")
