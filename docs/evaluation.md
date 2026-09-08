# Evaluation

Automated tests cover authorization, private-chat restrictions, input limits, subprocess cleanup, local-only model requests, malformed/truncated model output, transcript ownership, duplicate handling, bounded queues, restart recovery and expiry. Summary tests cover full-transcript input, language routing, excessive-length retry and compatibility with older cached excerpts. They use synthetic data and mocked network responses, not real Telegram accounts, and do not establish model accuracy.

A synthetic English/Russian smoke test checks that the installed Whisper and Qwen models run together. Synthetic speech is easier than noisy forwarded recordings and does not establish real-world accuracy or five-minute latency.

## Initial local checks

On an Apple M4 Mac with 16 GiB RAM, Whisper `large-v3-turbo` transcribed two short synthetic recordings (about 11 and 13 seconds) with the expected requests, amount, pending approval, negative instruction and future speaker commitment. With the final excerpt-selection implementation, transcription took about five seconds per clip; Qwen3.5 4B selection took about 13 seconds for the first clip and five seconds for the second. These are individual observations, affected by model loading and other activity, not a performance benchmark.

Earlier free-form summary experiments with both Qwen3.5 4B and 9B sometimes changed meaning: pending acceptance became rejection, or a future confirmation became an event date. The first release therefore selected exact source sentences. The synthetic checks of that release preserved every selected sentence's wording, including the negation and future commitment.

The personal pilot then exposed a usability problem: conversational transcripts can contain very long sentences or no punctuation, so selecting sentences can reproduce almost the entire message. The current implementation instead provides the full transcript to Qwen3.5 9B and generates one compact paragraph with controlled rephrasing. A scaled word target and one bounded shortening retry address verbosity. The prompt distinguishes pending decisions, speaker commitments and conditional offers, and asks the model to attribute claims rather than turn them into verified facts. These are quality measures, not factual guarantees.

Live checks produced substantially shorter paragraphs for longer recordings. Remaining errors included an already-completed action being described as a new task, an inferred causal link, and a speaker's commitment retaining first-person wording. The three initial pilot replies were manually reviewed and corrected before replacement; that human review is not an automatic service feature. Automated tests verify the application behavior, not the truth of a model-generated summary. A second model editing pass did not reliably fix the errors and was not added. The slower thinking-mode experiment timed out and was not enabled.

Speech recognition, summarization and translation can still introduce errors or omit context. Five-minute turnaround time and long-term sleep/restart reliability are not established by the short checks above.

## Personal pilot

Before relying on summaries, evaluate 10–15 representative recordings across:

| Case | What to check |
|---|---|
| English and Russian | Correct detected language and readable output |
| Names and company names | No invented spelling presented as certain |
| Amounts and dates | Exact values and units; no inferred calendar dates |
| Negation and tentative plans | Pending approval stays pending; do-not instructions remain negative |
| Background noise and silence | No fabricated transcript or summary |
| Short and five-minute notes | Full content processed within acceptable time |
| Summary model unavailable | Transcript remains available and retry works |
| Sleep/restart | Pending work resumes when the bot restarts |

Record duration, transcription time, summary time and material errors separately. Start with `large-v3-turbo`; compare `small` if memory or latency is a problem. Use the smallest model that preserves the details you need. Do not extrapolate from two short synthetic clips to a production accuracy claim.

No private pilot audio, transcripts, credentials or model weights belong in this public repository. Live Telegram delivery requires a dedicated BotFather token and an explicit allowed-user list.
