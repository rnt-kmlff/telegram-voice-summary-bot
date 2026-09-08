"""Download a canonical Whisper weight file and verify its published SHA-256."""

import argparse
import hashlib
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", choices=["small", "medium", "large-v3-turbo"], default="large-v3-turbo", nargs="?"
    )
    args = parser.parse_args()
    filename = f"ggml-{args.model}.bin"
    destination = Path("models") / filename
    destination.parent.mkdir(exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True, trust_env=False) as client:
        metadata = client.get(
            "https://huggingface.co/api/models/ggerganov/whisper.cpp/revision/main",
            params={"blobs": "true"},
        )
        metadata.raise_for_status()
        payload = metadata.json()
        revision = payload["sha"]
        entry = next(x for x in payload["siblings"] if x["rfilename"] == filename)
        expected = entry["lfs"]["sha256"]
        if destination.exists():
            with destination.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() == expected:
                    print(f"Verified existing {destination}")
                    return
            raise SystemExit(f"Existing {destination} has a different hash; move it aside first.")
        partial = destination.with_suffix(".bin.partial")
        try:
            digest = hashlib.sha256()
            received = 0
            next_report = 100_000_000
            url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{revision}/{filename}"
            with client.stream("GET", url) as response, partial.open("wb") as output:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if received >= next_report:
                        print(f"Downloaded {received // 1_000_000} MB", flush=True)
                        next_report += 100_000_000
            if received != entry["lfs"]["size"] or digest.hexdigest() != expected:
                raise SystemExit("Model integrity check failed; partial download removed.")
            partial.replace(destination)
            print(f"Verified {destination}; SHA-256 {expected}")
        finally:
            partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
