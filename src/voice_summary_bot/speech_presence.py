"""Mandatory local CPU speech-presence gate; no transcription, upload or audio trimming."""

import asyncio
import hashlib
import math
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .errors import NoClearSpeechError, SpeechCheckUnavailableError

SPEECH_CHECK_VERSION = 1
VAD_MODEL_FILENAME = "ggml-silero-v6.2.0.bin"
VAD_MODEL_REVISION = "9ffd54a1e1ee413ddf265af9913beaf518d1639b"
VAD_MODEL_SHA256 = "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"
VAD_MODEL_BYTES = 885_098
VAD_MODEL_URL = (
    f"https://huggingface.co/ggml-org/whisper-vad/resolve/{VAD_MODEL_REVISION}/{VAD_MODEL_FILENAME}"
)
_MAX_OUTPUT_BYTES = 262_144
_SEGMENT = re.compile(r"Speech segment (\d+): start = (\d+\.\d{2}), end = (\d+\.\d{2})")


def verify_vad_model(path: Path) -> None:
    """Require the reviewed model bytes, including when an alternate path is configured."""
    try:
        if path.stat().st_size != VAD_MODEL_BYTES:
            raise SpeechCheckUnavailableError()
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != VAD_MODEL_SHA256:
                raise SpeechCheckUnavailableError()
    except OSError:
        raise SpeechCheckUnavailableError() from None


@dataclass(frozen=True)
class SpeechPresence:
    audio_seconds: float
    speech_seconds: float

    @property
    def required_seconds(self) -> float:
        # Preserve short acknowledgements: below five seconds, require only 10%
        # of the clip. Longer clips need at least half a second of detected speech.
        return min(0.5, self.audio_seconds * 0.1)

    @property
    def sufficient(self) -> bool:
        return self.speech_seconds > 0 and self.speech_seconds + 1e-9 >= self.required_seconds


def parse_speech_presence(output: bytes, audio_seconds: float) -> SpeechPresence:
    """Parse the native tool's centiseconds, validating and merging overlapping spans."""
    try:
        if not math.isfinite(audio_seconds) or not 0 < audio_seconds <= 600:
            raise ValueError
        if len(output) > _MAX_OUTPUT_BYTES:
            raise ValueError
        lines = output.decode("utf-8").strip().splitlines()
        header = re.fullmatch(r"Detected (\d+) speech segments:", lines[0])
        if not header or int(header[1]) != len(lines) - 1:
            raise ValueError
        intervals = []
        for index, line in enumerate(lines[1:]):
            match = _SEGMENT.fullmatch(line)
            if not match or int(match[1]) != index:
                raise ValueError
            start, end = float(match[2]) / 100, float(match[3]) / 100
            if not 0 <= start < end <= audio_seconds + 0.05 or start >= audio_seconds:
                raise ValueError
            intervals.append((start, min(end, audio_seconds)))
        total, previous_end = 0.0, 0.0
        for start, end in sorted(intervals):
            total += max(0.0, end - max(start, previous_end))
            previous_end = max(previous_end, end)
        return SpeechPresence(audio_seconds, total)
    except (ValueError, IndexError, UnicodeError, OverflowError):
        raise SpeechCheckUnavailableError() from None


async def _capture_detector(*arguments: str, timeout: float) -> bytes:
    """Bound native output/time and reap the owned child after failure or cancellation."""
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        raise SpeechCheckUnavailableError() from None
    try:
        async with asyncio.timeout(timeout):
            output = bytearray()
            while chunk := await process.stdout.read(4096):
                output.extend(chunk)
                if len(output) > _MAX_OUTPUT_BYTES:
                    raise SpeechCheckUnavailableError()
            await process.wait()
        if process.returncode != 0:
            raise SpeechCheckUnavailableError()
        return bytes(output)
    except BaseException as error:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()
        if isinstance(error, (asyncio.CancelledError, SpeechCheckUnavailableError)):
            raise
        raise SpeechCheckUnavailableError() from None


class SpeechPresenceDetector:
    def __init__(
        self,
        model_path: Path = Path("models") / VAD_MODEL_FILENAME,
        binary: str = "whisper-vad-speech-segments",
        timeout: float = 30,
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Use a positive speech-check timeout.")
        self.model_path = Path(model_path).expanduser().resolve()
        self.binary = binary
        self.timeout = timeout

    async def check(self, wav_path: Path, audio_seconds: float) -> SpeechPresence:
        await asyncio.to_thread(verify_vad_model, self.model_path)
        output = await _capture_detector(
            self.binary,
            "-vm",
            str(self.model_path),
            "-f",
            str(wav_path),
            "-t",
            "2",
            "-vt",
            "0.5",
            "-vspd",
            "250",
            "-vsd",
            "100",
            "-vp",
            "30",
            "-vo",
            "0.1",
            "-np",
            timeout=self.timeout,
        )
        result = parse_speech_presence(output, audio_seconds)
        if not result.sufficient:
            raise NoClearSpeechError()
        return result
