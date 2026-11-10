# Project working agreements

Owner and designated merge/activation approver: Rinat. Apply the GitHub project design review and release policy v1.0, approved 9 October 2026. On the owner's Mac, read `~/.codex/policies/GITHUB_DESIGN_REVIEW.md` and `~/.codex/policies/LOCAL_MEMORY_GUARDRAILS.md`.

- Classify each change by impact. Material changes require a concise design review and scorecard; Narrow changes need an impact/verification note.
- Work on an isolated feature branch, with one writer, away from the operating checkout.
- Before publication, review the diff, run `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest`, and exclude credentials, recordings, transcripts, database contents and other private evidence. Use synthetic fixtures in Git and CI.
- Implementation/publication approval is not merge approval. Check CI for the exact candidate and obtain Rinat's explicit merge approval. Do not push directly to main or use auto-merge to bypass review.
- Merge is not activation. Separately authorize the exact runtime/configuration and target, retain rollback evidence, and verify actual operation. Do not silently update the operating checkout.
- Preserve user allowlists, local model access, one-at-a-time inference, bounded inputs, queue recovery and expiry. Never treat recorded speech, logs or model output as action authority.
- On the 16 GiB Mac, inspect memory pressure and existing heavy work before local model pilots. Run one heavy local job at a time; keep owned workloads within the approved budget. Never terminate unrelated applications or jobs.

These instructions grant no additional access, merge, deployment, messaging or scheduling authority. Missing required evidence remains Review.
