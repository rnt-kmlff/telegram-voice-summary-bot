"""Fetch the reviewed, checksum-pinned Silero speech detector; never fetch at runtime."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from voice_summary_bot.speech_presence import (
    VAD_MODEL_BYTES,
    VAD_MODEL_FILENAME,
    VAD_MODEL_URL,
    verify_vad_model,
)


def download(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / VAD_MODEL_FILENAME
    if destination.exists():
        verify_vad_model(destination)
        return destination
    with NamedTemporaryFile(prefix=".vad-", dir=directory, delete=False) as output:
        temporary = Path(output.name)
        try:
            with (
                httpx.Client(timeout=60, follow_redirects=True, trust_env=False) as client,
                client.stream("GET", VAD_MODEL_URL) as response,
            ):
                response.raise_for_status()
                received = 0
                for chunk in response.iter_bytes(65536):
                    received += len(chunk)
                    if received > VAD_MODEL_BYTES:
                        raise ValueError("Speech model exceeds its reviewed size.")
                    output.write(chunk)
            output.flush()
            verify_vad_model(temporary)
            if destination.exists():
                verify_vad_model(destination)
            else:
                temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("models"))
    args = parser.parse_args()
    try:
        path = download(args.directory)
    except Exception as exc:
        raise SystemExit(f"Speech model download failed ({type(exc).__name__}).") from None
    print(f"Verified {path}")
