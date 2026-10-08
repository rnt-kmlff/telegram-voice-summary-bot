"""Synthetic boundaries: remove distinctive endings without global text replacement."""

import json

import pytest

from voice_summary_bot.inference import _parse_whisper_json
from voice_summary_bot.transcript_cleanup import filter_whisper_endings


@pytest.mark.parametrize(
    "credit",
    [
        "Субтитры сделал DimaTorzok",
        "Субтитры создавал DimaTorzok",
        "СУБТИТРЫ СДЕЛАЛ dimatorzok!!!",
        "  Субтитры\nсоздавал:\u00a0DimaTorzok…  ",
    ],
)
def test_complete_trailing_credit_is_removed_with_exact_audit(credit):
    cleaned, removed = filter_whisper_endings(["Please confirm on Friday.", credit, " "])
    assert cleaned == "Please confirm on Friday."
    assert removed == (credit.strip(),)


def test_split_and_repeated_credits_are_recorded_in_source_order():
    cleaned, removed = filter_whisper_endings(
        ["Встреча в пятницу.", "Субтитры", "сделал", "DimaTorzok", "Субтитры создавал DimaTorzok."]
    )
    assert cleaned == "Встреча в пятницу."
    assert removed == ("Субтитры сделал DimaTorzok", "Субтитры создавал DimaTorzok.")


@pytest.mark.parametrize(
    "segments",
    [
        ["Субтитры сделал DimaTorzok", "А теперь обсудим встречу."],
        ["В конце было написано: Субтитры сделал DimaTorzok"],
        ["Встреча в пятницу. Субтитры сделал DimaTorzok"],
        ['"Субтитры сделал DimaTorzok"'],
        ["«Субтитры сделал DimaTorzok»"],
        ["Субтитры сделал другой человек."],
        ["Субтитры сделал DimaTorzokOfficial"],
        ["Субтитры не сделал DimaTorzok"],
        ["Неизвестная заключительная фраза."],
        ["Спасибо за просмотр!"],
        ["Спасибо."],
        ["Продолжение следует…"],
        ["Thank you for watching."],
        ["  A pause. ", "", "Continue.", " "],
    ],
)
def test_ordinary_quoted_embedded_and_nonterminal_text_is_preserved(segments):
    cleaned, removed = filter_whisper_endings(segments)
    assert cleaned == " ".join(segment.strip() for segment in segments).strip()
    assert removed == ()


def test_ordinary_ending_stops_filter_before_an_earlier_credit():
    source = ["Субтитры сделал DimaTorzok", "Спасибо!", "Субтитры создавал DimaTorzok"]
    assert filter_whisper_endings(source) == (
        "Субтитры сделал DimaTorzok Спасибо!",
        ("Субтитры создавал DimaTorzok",),
    )


def test_parser_keeps_unfiltered_text_and_removal_record(tmp_path):
    path = tmp_path / "whisper.json"
    source = ["Не подтверждайте пока.", "Субтитры сделал DimaTorzok"]
    path.write_text(
        json.dumps(
            {
                "result": {"language": "ru"},
                "transcription": [
                    {"text": text, "offsets": {"from": 0, "to": 999999}} for text in source
                ],
            }
        )
    )
    transcript = _parse_whisper_json(path, "auto")
    assert transcript.text == source[0]
    assert transcript.raw_text == " ".join(source)
    assert transcript.removed_endings == (source[1],)
    assert transcript.language == "ru"
    assert "DimaTorzok" not in repr(transcript)


def test_credit_only_audio_retains_audit_with_empty_cleaned_text(tmp_path):
    path = tmp_path / "whisper.json"
    credit = "Субтитры сделал DimaTorzok"
    path.write_text(json.dumps({"transcription": [{"text": credit}]}))
    transcript = _parse_whisper_json(path, "ru")
    assert transcript.text == ""
    assert transcript.raw_text == credit
    assert transcript.removed_endings == (credit,)


def test_bad_timestamps_alone_do_not_remove_real_words(tmp_path):
    path = tmp_path / "whisper.json"
    text = "Do not approve the plan."
    path.write_text(
        json.dumps({"transcription": [{"text": text, "offsets": {"from": 98020, "to": 128000}}]})
    )
    transcript = _parse_whisper_json(path, "en")
    assert transcript.text == text
    assert transcript.raw_text is None
    assert transcript.removed_endings == ()
