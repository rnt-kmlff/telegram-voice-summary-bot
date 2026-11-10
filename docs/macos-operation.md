# Running on a Mac

Keep the repository in a local folder such as `~/Projects`, outside iCloud Drive, OneDrive or other synced folders. A Python environment contains thousands of small files that a sync service can mark hidden, duplicate or offload. Keep the local model files outside synced storage too.

Start Ollama automatically at login if desired:

```sh
brew services start ollama
```

Ollama normally listens on loopback. Keep it there; this project checks for loopback URLs and downloaded local models. Do not expose the Ollama port to a public network.

After creating `.env` and setting the token and allowlist, check dependencies from the repository:

```sh
uv run voice-summary doctor
```

## Optional bot login service

Stop any manually running copy of the bot with Control-C before installing the service. From the repository, run:

```sh
uv run python scripts/macos_service.py install
uv run python scripts/macos_service.py status
```

The installer checks the private configuration, Python environment, Whisper model, checksum-pinned speech detector model and required commands. It creates a user login service named `com.telegram-voice-summary-bot.local`, starts it now, and starts it again when you log in. macOS restarts the process after an exit, with a 30-second throttle. This is a login service, so you must be signed in; it does not keep a sleeping Mac awake.

The service runs this checkout's `.venv` Python directly. The token and allowlist stay in `.env`; their values are not written to the service definition or its command. The definition is private to the current account, and output goes to `data/bot-service.log` with private file permissions. The service has a fixed Homebrew/system command search path. If your binaries are installed elsewhere, set their absolute paths in `.env` before installing.

Run `install` again after changing `.env` to restart this checkout's service. The helper refuses to replace or remove a service belonging to a different checkout. Uninstall it from the original checkout before moving the repository. To stop automatic operation and remove the service:

```sh
uv run python scripts/macos_service.py uninstall
```

Uninstalling keeps `.env`, queued work, results, logs and models. It leaves Ollama running independently. This helper does not rotate logs; review their size periodically. Never run the manual bot and login service together for the same token.

## Manual operation

When the bot login service is uninstalled, run:

```sh
uv run voice-summary bot
```

Leave that terminal open. To keep the Mac awake during an intentional processing session:

```sh
caffeinate -i uv run voice-summary bot
```

This does not promise processing with a closed laptop lid. Stop with Control-C. The bot saves pending jobs and releases its process lock. Start it again from the same directory to resume. Run only one instance per bot token.

Ollama can keep running independently for other applications. Stop its service with `brew services stop ollama` when it is no longer needed by any application.

## Troubleshooting

- **No replies:** confirm the Mac is awake, the bot terminal or login service is running, your numeric ID is allowlisted, and you are messaging the dedicated bot in a private chat. The service `status` command reports the process state; it does not verify Telegram delivery or model readiness.
- **Processing stopped:** restart the bot and check available disk space and write access to its `data` directory. For the login service, run its `install` command again. The bot refuses new inference work if its worker has failed; a living process with a failed worker may require this explicit restart.
- **No clear speech detected:** listen to the original recording. Resend a clearer recording or send its content as text.
- **Speech check unavailable:** run `uv run voice-summary doctor`; check the `whisper-vad-speech-segments` command and install the pinned model with `uv run python scripts/download_vad.py`. An existing incorrect model is preserved rather than overwritten; inspect and move it aside before downloading the reviewed model. Follow the owner-approved release procedure before changing an operating checkout.
- **Summary unavailable:** use Full transcript; check Ollama and `ollama list`, then forward the message again.
- **Command cannot import its package:** use a local, unsynced checkout and recreate its generated `.venv`. A regular package installation (`uv sync --no-editable`, then `.venv/bin/voice-summary bot`) is an alternative when a platform marks editable `.pth` files hidden.
- **Model is slow:** other model clients compete for the same memory and GPU. Close unnecessary model sessions or compare Whisper `small` on representative recordings.
- **Language setting changed:** completed duplicate recordings with the current speech-check version reuse their cached summary until it expires. `/language` applies to newly processed recordings.

Audio temporary directories are cleaned after normal processing. A forced process kill may leave audio files until the next bot startup; startup cleans only its generated directories under `data/audio-tmp` while holding the process lock. Standalone audio CLI commands use operating-system temporary storage and may leave temporary files after a hard crash. Backups and Telegram copies have separate retention.

Full transcript downloads include the complete summary when one is available. If a summary exceeds Telegram's message size, the chat shows an explicit preview notice; the download preserves all fields.
