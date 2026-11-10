"""Public-safe inference errors; never include source audio or native tool output."""


class InferenceError(Exception):
    """An inference failure whose message is safe to show without private content."""


class AudioDurationError(InferenceError):
    """The decoded audio exceeds the recording's accepted limit."""


class NoClearSpeechError(InferenceError):
    def __init__(self) -> None:
        super().__init__(
            "No clear speech detected. Please resend the recording or send the message as text."
        )


class SpeechCheckUnavailableError(InferenceError):
    def __init__(self) -> None:
        super().__init__("The speech check is unavailable. Please try again later.")
