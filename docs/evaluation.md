# Evaluation

Automated tests cover authorization, private-chat restrictions, input limits, subprocess cleanup, local-only model requests, malformed/truncated model output, transcript ownership, duplicate handling, bounded queues, restart recovery and expiry. Duration tests cover persistent per-user choices, migration of existing jobs, fixed limits for queued recordings, deployment ceilings, command buttons, and decoded WAV boundaries at exactly 300/600 seconds and one frame over. Summary tests cover full-transcript input, language routing, excessive-length retry and compatibility with older cached excerpts. They use synthetic data and mocked network responses, not real Telegram accounts, and do not establish model accuracy.

A synthetic English/Russian smoke test checks that the installed Whisper and Qwen models run together. Synthetic speech is easier than noisy forwarded recordings and does not establish real-world accuracy.

## Initial local checks

On an Apple M4 Mac with 16 GiB RAM, Whisper `large-v3-turbo` transcribed two short synthetic recordings (about 11 and 13 seconds) with the expected requests, amount, pending approval, negative instruction and future speaker commitment. With the final excerpt-selection implementation, transcription took about five seconds per clip; Qwen3.5 4B selection took about 13 seconds for the first clip and five seconds for the second. These are individual observations, affected by model loading and other activity, not a performance benchmark.

Earlier free-form summary experiments with both Qwen3.5 4B and 9B sometimes changed meaning: pending acceptance became rejection, or a future confirmation became an event date. The first release therefore selected exact source sentences. The synthetic checks of that release preserved every selected sentence's wording, including the negation and future commitment.

The personal pilot then exposed a usability problem: conversational transcripts can contain very long sentences or no punctuation, so selecting sentences can reproduce almost the entire message. The current implementation instead provides the full transcript to Qwen3.5 9B and generates one compact paragraph with controlled rephrasing. A scaled word target and one bounded shortening retry address verbosity. The prompt distinguishes pending decisions, speaker commitments and conditional offers, and asks the model to attribute claims rather than turn them into verified facts. These are quality measures, not factual guarantees.

Live checks produced substantially shorter paragraphs for longer recordings. Remaining errors included an already-completed action being described as a new task, an inferred causal link, and a speaker's commitment retaining first-person wording. The three initial pilot replies were manually reviewed and corrected before replacement; that human review is not an automatic service feature. Automated tests verify the application behavior, not the truth of a model-generated summary. A second model editing pass did not reliably fix the errors and was not added. The slower thinking-mode experiment timed out and was not enabled.

Speech recognition, summarization and translation can still introduce errors or omit context. Long-recording turnaround time and long-term sleep/restart reliability are not established by the short checks above.

## Ten-minute local checks — 8 September 2026

Two synthetic workshop updates, one English and one Russian, were generated with macOS speech synthesis and normalized to exactly 600 seconds. Each contained 20 similar sections with distinct checkpoints at the beginning, middle and end, plus a pending second-room offer and final instructions about dates and booking confirmation. The Opus files were about 3.3–3.4 MB each.

On the same Apple M4 / 16 GiB Mac, using Whisper `large-v3-turbo` and Qwen3.5 9B with a 16,384-token context and thinking disabled:

| Recording | Transcription | Summary | Local pipeline total |
|---|---:|---:|---:|
| English, 10 minutes | 60.48 s | 37.78 s | 98.26 s |
| Russian, 10 minutes | 68.92 s | 37.94 s | 106.86 s |

Both transcripts contained all three checkpoints, including the final one. Both summaries condensed the repeated sections into a paragraph and retained the pending room offer, Friday plan, Thursday-evening question deadline and instruction to wait for booking confirmation before publishing invitations. The same ten-minute English file was rejected in five-minute mode after decoding, before Whisper ran (0.42 seconds).

These are individual local runs, excluding Telegram transfer and queue waiting. Repetitive synthetic speech is easier to summarize than a recording with many unrelated topics; the checks establish operation at ten minutes, not a general accuracy or latency guarantee. The deployed service's command menu and persisted ten-minute preference were read back separately through Telegram and local storage.

## Whisper ending cleanup — 8 October 2026

A privately supplied Russian recording reproduced the distinctive `Субтитры сделал DimaTorzok` ending in a separate final Whisper segment. With the literal ending filter enabled, the unfiltered output matched the previously exported transcript exactly; the cleaned text matched it exactly after removing only that credit. The local audit was saved and read back successfully. Temporary diagnostic copies were removed, and no private recording or transcript was added to this repository.

Synthetic regression tests cover case/punctuation variants, credits split across final segments, repeated credits, ordinary closings, quoted/embedded/mid-transcript mentions, invalid timestamps accompanying genuine words, and recordings containing only a credit. Integration tests verify that only cleaned text reaches summarization and Telegram exports, while the local audit survives restart/summary failure and is erased with the terminal job at expiry. Package validation also checks that the editable list is included in the wheel. These checks validate this narrow filter; they do not establish detection of other hallucinations.

## Personal pilot

Before relying on summaries, evaluate 10–15 representative recordings across:

| Case | What to check |
|---|---|
| English and Russian | Correct detected language and readable output |
| Names and company names | No invented spelling presented as certain |
| Amounts and dates | Exact values and units; no inferred calendar dates |
| Negation and tentative plans | Pending approval stays pending; do-not instructions remain negative |
| Background noise and silence | No fabricated transcript or summary |
| Short, five-minute and ten-minute notes | Full content processed within acceptable time |
| Summary model unavailable | Transcript remains available and retry works |
| Sleep/restart | Pending work resumes when the bot restarts |

Record duration, transcription time, summary time and material errors separately. Start with `large-v3-turbo`; compare `small` if memory or latency is a problem. Use the smallest model that preserves the details you need. Do not extrapolate from two short synthetic clips to a production accuracy claim.

No private pilot audio, transcripts, credentials or model weights belong in this public repository. Live Telegram delivery requires a dedicated BotFather token and an explicit allowed-user list.

## Speech-presence gate — 10 November 2026

An isolated local pilot used whisper.cpp 1.9.2's CPU detector with the pinned Silero v6.2.0 model. Each clip ran sequentially; no Telegram delivery or Qwen request was involved. These are synthetic smoke checks, not estimates of real-world accuracy.

| Synthetic sample | Duration | Detected speech | Result |
| --- | ---: | ---: | --- |
| English request, macOS Samantha | 4.108 s | 4.108 s | Accepted |
| Russian request, macOS Milena | 6.264 s | 6.064 s | Accepted |
| English “Yes” | 0.481 s | 0.481 s | Accepted |
| Russian “Да” | 0.378 s | 0.348 s | Accepted |
| English request at 8% amplitude | 4.108 s | 4.108 s | Accepted |
| English request with 5 s leading and 10 s trailing silence | 19.108 s | 4.160 s | Accepted |
| Digital silence | 15 s | 0 s | Rejected |
| Alternating impulses every half second | 15 s | 0 s | Rejected |
| Ten one-minute blocks, each starting with the English request | 600 s | 41.300 s | Accepted |

Synthetic requests: “Please move our meeting to Friday at ten. I will send the revised document tomorrow.” and “Давай перенесём встречу на пятницу, на десять часов. Завтра я отправлю исправленный документ.” Speech was converted to mono 16 kHz signed 16-bit PCM. These generated files and model weights are not committed. The ten-minute CPU check took 0.923 seconds; the native child peaked at about 110 MiB RSS, measured by the operating system after exit. Memory pressure was green before and after the serial pilot. These are single-run observations, not performance guarantees.

Automated regressions cover native centisecond units, threshold boundaries, overlapping spans, malformed/truncated/oversized output, missing/corrupt model, bounded downloads, timeout/cancellation cleanup, and no Whisper/Qwen call or transcript button after rejection. They also cover checked versus unchecked duplicate results, migration, capacity, restart and expiry. A byte-for-byte assertion verifies that successful detection does not trim the WAV sent to Whisper.

Limitations: this gate detects speech-like activity, not intelligibility or transcription accuracy. Brief real speech in a long recording may be rejected; sufficiently speech-like noise may pass. Live Telegram delivery and operation after deployment require a separate activation check. The summary model and its prompt are unchanged, so a new real-Qwen inference pilot was not run for this change.
