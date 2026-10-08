"""Conservative, literal-list cleanup of complete trailing Whisper segments."""

import unicodedata
from functools import cache
from importlib.resources import files

_PUNCTUATION = str.maketrans({mark: " " for mark in ".,:;!?…"})


def _normalize(text: str) -> str:
    # Do not strip quotation marks: a quoted credit can be intentional speech.
    return " ".join(unicodedata.normalize("NFC", text).casefold().translate(_PUNCTUATION).split())


@cache
def _known_endings() -> frozenset[str]:
    text = files("voice_summary_bot").joinpath("whisper_endings.txt").read_text(encoding="utf-8")
    return frozenset(
        _normalize(line)
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def filter_whisper_endings(segments: list[str]) -> tuple[str, tuple[str, ...]]:
    """Return cleaned text and exact removed credits in their original order.

    Only complete trailing segments can be removed. A credit split across several
    terminal segments also matches; an ordinary speech segment stops the scan.
    Timestamps are not used to decide whether text is genuine.
    """
    raw_text = " ".join(segment.strip() for segment in segments).strip()
    parts = [(index, segment.strip()) for index, segment in enumerate(segments) if segment.strip()]
    endings = _known_endings()
    max_span = max((len(ending.split()) for ending in endings), default=0)
    removed: list[str] = []
    cut_at = len(segments)
    while parts:
        for span in range(min(max_span, len(parts)), 0, -1):
            candidate = " ".join(text for _, text in parts[-span:])
            if _normalize(candidate) in endings:
                cut_at = parts[-span][0]
                removed.append(candidate)
                del parts[-span:]
                break
        else:
            break
    if not removed:
        return raw_text, ()
    cleaned = " ".join(segment.strip() for segment in segments[:cut_at]).strip()
    return cleaned, tuple(reversed(removed))
