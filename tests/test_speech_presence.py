"""Synthetic detector protocol, rejection and process-cleanup regressions."""

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from voice_summary_bot import speech_presence as speech
from voice_summary_bot.errors import NoClearSpeechError, SpeechCheckUnavailableError


def native(*intervals):
    lines = [f"Detected {len(intervals)} speech segments:"]
    lines += [
        f"Speech segment {index}: start = {start:.2f}, end = {end:.2f}"
        for index, (start, end) in enumerate(intervals)
    ]
    return ("\n" + "\n".join(lines) + "\n").encode()


@pytest.mark.parametrize(
    "duration,intervals,seconds,accepted",
    [
        (60, [(80, 112)], 0.32, False),
        (60, [(80, 130)], 0.5, True),
        (600, [(100, 200)], 1, True),
        (2, [(80, 112)], 0.32, True),
        (5, [(0, 49)], 0.49, False),
        (5, [(0, 50)], 0.5, True),
        (5, [], 0, False),
        # Overlap must not double-count a short click as enough speech.
        (10, [(0, 30), (10, 40)], 0.4, False),
        (1, [(80, 103)], 0.2, True),
    ],
)
def test_native_centiseconds_and_conservative_threshold(duration, intervals, seconds, accepted):
    result = speech.parse_speech_presence(native(*intervals), duration)
    assert result.speech_seconds == pytest.approx(seconds)
    assert result.sufficient is accepted


@pytest.mark.parametrize(
    "payload,duration",
    [
        (b"", 5),
        (b"private error details", 5),
        (b"\xff", 5),
        (b"Detected 1 speech segments:\n", 5),
        (b"Detected 0 speech segments:\nextra", 5),
        (b"Detected 1 speech segments:\nSpeech segment 1: start = 0.00, end = 50.00", 5),
        (b"Detected 1 speech segments:\nSpeech segment 0: start = nan, end = 50.00", 5),
        (native((-1, 50)), 5),
        (native((50, 50)), 5),
        (native((500, 550)), 5),
        (native((0, 506)), 5),
        (native(), float("nan")),
        (native(), 601),
        (native(), 0),
        (b"x" * (speech._MAX_OUTPUT_BYTES + 1), 5),
    ],
)
def test_malformed_or_unbounded_results_fail_closed(payload, duration):
    with pytest.raises(SpeechCheckUnavailableError) as caught:
        speech.parse_speech_presence(payload, duration)
    assert str(caught.value) == "The speech check is unavailable. Please try again later."


def test_model_requires_both_reviewed_size_and_hash(tmp_path, monkeypatch):
    path = tmp_path / "model.bin"
    approved = b"synthetic reviewed bytes"
    monkeypatch.setattr(speech, "VAD_MODEL_BYTES", len(approved))
    monkeypatch.setattr(speech, "VAD_MODEL_SHA256", hashlib.sha256(approved).hexdigest())
    with pytest.raises(SpeechCheckUnavailableError):
        speech.verify_vad_model(path)
    path.write_bytes(approved)
    speech.verify_vad_model(path)
    path.write_bytes(b"x" * len(approved))
    with pytest.raises(SpeechCheckUnavailableError):
        speech.verify_vad_model(path)
    path.write_bytes(approved + b"x")
    with pytest.raises(SpeechCheckUnavailableError):
        speech.verify_vad_model(path)


@pytest.mark.parametrize("intervals,accepted", [([], False), ([(0, 32)], False), ([(0, 60)], True)])
async def test_detector_cpu_and_rejection(monkeypatch, tmp_path, intervals, accepted):
    monkeypatch.setattr(speech, "verify_vad_model", lambda _: None)
    capture = AsyncMock(return_value=native(*intervals))
    monkeypatch.setattr(speech, "_capture_detector", capture)
    detector = speech.SpeechPresenceDetector(tmp_path / "model")
    if accepted:
        assert (await detector.check(tmp_path / "audio.wav", 60)).sufficient
    else:
        with pytest.raises(NoClearSpeechError, match="Please resend"):
            await detector.check(tmp_path / "audio.wav", 60)
    arguments = capture.call_args.args
    assert "-ug" not in arguments
    assert arguments[arguments.index("-t") + 1] == "2"
    assert capture.call_args.kwargs == {"timeout": 30}


async def test_unverified_model_does_not_start_a_process(monkeypatch, tmp_path):
    capture = AsyncMock()
    monkeypatch.setattr(speech, "_capture_detector", capture)
    with pytest.raises(SpeechCheckUnavailableError):
        await speech.SpeechPresenceDetector(tmp_path / "missing").check(Path("audio.wav"), 3)
    capture.assert_not_awaited()


@pytest.mark.parametrize("mode", ["timeout", "cancel", "oversize", "exit", "success"])
async def test_subprocess_is_bounded_and_reaped(monkeypatch, mode):
    entered = asyncio.Event()

    class Process:
        returncode = None
        killed = False
        waits = 0
        reads = 0

        def __init__(self):
            self.stdout = self

        async def read(self, count):
            entered.set()
            self.reads += 1
            if mode in {"timeout", "cancel"}:
                await asyncio.Event().wait()
            if mode == "oversize":
                return b"x" * count
            return b"safe output" if self.reads == 1 else b""

        async def wait(self):
            self.waits += 1
            if self.returncode is None:
                self.returncode = 1 if mode == "exit" else 0
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = Process()
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    task = asyncio.create_task(speech._capture_detector("detector", "literal;name", timeout=0.01))
    if mode == "cancel":
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif mode == "success":
        assert await task == b"safe output"
    else:
        with pytest.raises(SpeechCheckUnavailableError):
            await task
    assert process.waits >= 1
    assert process.killed is (mode in {"timeout", "cancel", "oversize"})
    assert create.call_args.args == ("detector", "literal;name")
    assert create.call_args.kwargs["stderr"] == asyncio.subprocess.DEVNULL


async def test_missing_binary_is_an_operational_error(monkeypatch):
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError))
    with pytest.raises(SpeechCheckUnavailableError):
        await speech._capture_detector("missing", timeout=1)
