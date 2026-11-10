# Speech-presence gate: design review

**Impact: Material. Owner and merge/activation approver: Rinat.** The new model dependency, audio acceptance rule and cache schema affect processing and stored-result meaning. Candidate identity and exact CI results belong in the associated PR; this document is not merge or activation approval.

## Decision and scope

Require a local speech check after bounded audio decoding and duration validation, before Whisper and Qwen. Too little detected speech produces a resend/text request. Detector failure produces an operational error and never permits unchecked inference. Keep the existing 5/10-minute choices, allowlist, queue limits, local retention, full-audio transcription, ending cleanup and summary prompt.

A phrase blacklist alone cannot distinguish genuine words from hallucinations over noise. Peak volume alone accepts clicks. Passing VAD spans into Whisper would change which audio gets transcribed. This change uses a conservative presence gate instead, preserving the full accepted WAV. Threshold version 1 requires `min(0.5 seconds, duration * 0.1)` of detected speech. False acceptance can still yield invented text; false rejection costs a resend. Broader accuracy measurement remains ongoing personal evaluation, not a release accuracy claim.

## Dependency and resource contract

- CPU `whisper-vad-speech-segments`, two threads, 30-second timeout, 256 KiB output cap; child is killed and reaped on timeout, cancellation or excessive output. No GPU flag. Native timestamps are centiseconds; intervals are validated, clamped to decoded duration and merged before counting.
- Silero `ggml-silero-v6.2.0.bin`, 885,098 bytes, from [whisper-vad revision 9ffd54a](https://huggingface.co/ggml-org/whisper-vad/tree/9ffd54a1e1ee413ddf265af9913beaf518d1639b).
- SHA-256: `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987`.
- Explicit setup download only, pinned URL, bounded stream and verified temporary file before replacement. An incorrect existing model is not overwritten. Runtime checks model size and hash, and does not fetch weights.
- Default native options: threshold 0.5, minimum speech 250 ms, silence 100 ms, padding 30 ms, overlap 0.1 s. There is no configurable bypass. The installed native binary remains an operator-managed dependency; unexpected output fails closed.

## Integration, privacy and recovery

Telegram audio and standalone audio commands use the same gate. Text-only summarization does not require it. Audio is processed locally and deleted under existing temporary-file rules. Only timing output is captured; native diagnostics are discarded and no audio, detector output, source text or credentials are logged.

`jobs.speech_check_version` is an additive integer column, default 0. Version 1 is recorded with a successful transcript. A completed duplicate with another version is requeued, respecting capacity and the newly accepted duration limit, and its prior transcript/summary/audit is cleared. Current checked results can be reused; pending duplicates keep their accepted limits. Interrupted jobs retain existing restart recovery. A speech-check failure clears any stale transcript, summary and audit from a recovered attempt, and remains retryable. Terminal-job TTL remains unchanged. Previously exported Telegram files are outside this cache transition.

Rollback requires stopping the owned service at a safe boundary, retaining its private configuration and consistent database backup, restoring the previous approved code/runtime, and restarting only that service. The added column is compatible with the previous code, which selects known job fields. Do not drop it or restore an old database over newer jobs as an automatic rollback. Reverting removes this protection; the owner must explicitly accept that behavior. A later reactivation still checks cached-result versions.

## Scorecard and next gates

| Mandatory criterion | Status | Evidence / limit |
| --- | --- | --- |
| Scope, owner and cost of errors | Met | Defined above; detector is not an accuracy guarantee |
| Deterministic model and bounded processing | Met | Pinned size/hash, CPU options, timeout/output/process-cleanup tests |
| Integration and error routing | Met | Pipeline and bot tests prove rejection before Whisper/Qwen and no transcript artifact |
| Cache identity, migration, replay and capacity | Met | Versioned cache tests, additive migration, retry/capacity/restart/TTL checks |
| Privacy and unchanged AI authority | Met | Local gate; synthetic public evidence only; no new tools, destinations or model authority |
| Representative speech quality | Partial | English/Russian, short/quiet/paused synthetic smoke checks pass; broader real-world accuracy is not established |
| Local regression and package checks | Met | 265 tests, Ruff lint/format, diff checks and wheel contents verified; commands in PR |
| Exact-head CI | Review | Record final candidate SHA and CI result in PR before requesting merge |
| Merge approval and exact live activation | Review | Rinat must approve the candidate revision and operating target separately |
| Live delivery after activation | Review | Requires an authorized forwarded recording and observed bot response |

Ready for publication for owner review after the PR's regression and privacy checks pass. Not yet approved for merge or live activation. No scheduler or new background monitor is introduced. Safe failure messages and existing job state support diagnosis; threshold/model changes require another affected review.
