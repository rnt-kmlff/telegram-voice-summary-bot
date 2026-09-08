# Evaluation

Automated tests cover authorization, private-chat restrictions, input limits, subprocess cleanup, local-only model requests, malformed/truncated model output, transcript ownership, duplicate handling, bounded queues, restart recovery and expiry. They use synthetic data and mocked network responses, not real Telegram accounts.

A synthetic English/Russian smoke test checks that the installed Whisper and Qwen models run together. Synthetic speech is easier than noisy forwarded recordings and does not establish real-world accuracy or five-minute latency.

## Initial local checks

On an Apple M4 Mac with 16 GiB RAM, Whisper `large-v3-turbo` transcribed two short synthetic recordings (about 11 and 13 seconds) with the expected requests, amount, pending approval, negative instruction and future speaker commitment. With the final excerpt-selection implementation, transcription took about five seconds per clip; Qwen3.5 4B selection took about 13 seconds for the first clip and five seconds for the second. These are individual observations, affected by model loading and other activity, not a performance benchmark.

Earlier free-form summary experiments with both Qwen3.5 4B and 9B sometimes changed meaning: pending acceptance became rejection, or a future confirmation became an event date. Stronger prompts did not reliably eliminate these errors. The default therefore asks Qwen to select numbered source sentences and maps valid selections back to the original text in code. Both final synthetic checks preserved every selected sentence's wording, including the negation and future commitment.

Selection can still omit relevant context or put a sentence in the wrong category. Speech recognition can still introduce errors. Explicit cross-language output uses an additional machine-translation step and does not have the same verbatim guarantee. No real-user recording accuracy, five-minute turnaround time or long-term sleep/restart reliability is established by these checks.

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
