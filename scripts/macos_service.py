"""Install, inspect, or remove this checkout's private macOS login service."""

import argparse
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from voice_summary_bot.config import ConfigurationError, Settings

LABEL = "com.telegram-voice-summary-bot.local"
SERVICE_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"


class ServiceError(Exception):
    """An operational message that contains no configuration values."""


def build_plist(repo: Path) -> dict:
    repo = repo.resolve()
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(repo / ".venv" / "bin" / "python"),
            "-m",
            "voice_summary_bot.cli",
            "--env-file",
            str(repo / ".env"),
            "bot",
        ],
        "WorkingDirectory": str(repo),
        "EnvironmentVariables": {"PATH": SERVICE_PATH},
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 30,
        "StandardOutPath": str(repo / "data" / "bot-service.log"),
        "StandardErrorPath": str(repo / "data" / "bot-service.log"),
        "Umask": 0o077,
    }


def validate_install(repo: Path) -> None:
    python = repo / ".venv" / "bin" / "python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ServiceError("The checkout's Python environment is missing; run uv sync first.")
    if not (repo / ".env").is_file():
        raise ServiceError("Create the private .env file before installing the service.")
    # Match the service's working directory and configuration, without relying on
    # credentials or configuration exported in the installer's interactive shell.
    old_directory = Path.cwd()
    old_environment = os.environ.copy()
    try:
        os.chdir(repo)
        os.environ.clear()
        os.environ["PATH"] = SERVICE_PATH
        settings = Settings.load(repo / ".env", require_bot=True)
    finally:
        os.chdir(old_directory)
        os.environ.clear()
        os.environ.update(old_environment)
    if not settings.model_path.is_file():
        raise ServiceError("The configured Whisper model is missing; download it first.")
    for label, command in (
        ("Whisper", settings.whisper_binary),
        ("FFmpeg", settings.ffmpeg_binary),
    ):
        if shutil.which(command, path=SERVICE_PATH) is None:
            raise ServiceError(f"The configured {label} command is unavailable to the service.")


def existing_plist(path: Path, expected: dict) -> dict | None:
    if path.is_symlink():
        raise ServiceError("The service definition is a symbolic link; refusing to change it.")
    if not path.exists():
        return None
    try:
        with path.open("rb") as source:
            existing = plistlib.load(source)
    except (OSError, ValueError, plistlib.InvalidFileException):
        raise ServiceError("The existing service definition could not be verified.") from None
    if not isinstance(existing, dict) or any(
        existing.get(key) != expected[key]
        for key in ("Label", "WorkingDirectory", "ProgramArguments")
    ):
        raise ServiceError(
            "This service label belongs to a different checkout or command. "
            "Remove it from its original checkout before continuing."
        )
    return existing


def run_launchctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            ["/bin/launchctl", *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            # Settings.load reads .env; never pass its values to launchctl.
            env={"PATH": SERVICE_PATH},
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ServiceError("The macOS service manager could not be reached.") from None
    if check and result.returncode:
        raise ServiceError(
            f"The macOS service operation failed (exit {result.returncode}). "
            "Check that you are signed into the desktop, then retry."
        )
    return result


def _bootout(service: str) -> None:
    result = run_launchctl("bootout", service, check=False)
    if result.returncode not in {0, 3}:
        raise ServiceError("The previous service could not be stopped; no replacement was started.")


def _private_log(repo: Path) -> None:
    directory = repo / "data"
    if directory.is_symlink():
        raise ServiceError("The service log directory must not be a symbolic link.")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    flags = os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory / "bot-service.log", flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _write_plist(path: Path, definition: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{LABEL}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            os.fchmod(destination.fileno(), 0o600)
            plistlib.dump(definition, destination, fmt=plistlib.FMT_XML, sort_keys=True)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install(repo: Path, agents_directory: Path) -> None:
    definition = build_plist(repo)
    destination = agents_directory / f"{LABEL}.plist"
    previous = existing_plist(destination, definition)
    validate_install(repo)
    _private_log(repo)
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/{LABEL}"
    if previous is not None:
        _bootout(service)
    _write_plist(destination, definition)
    # Enable first: a previously disabled label must be eligible for RunAtLoad
    # when it is bootstrapped. No token or user ID is passed to launchd.
    run_launchctl("enable", service)
    run_launchctl("bootstrap", domain, str(destination))
    print("Bot login service installed. It starts at login and restarts after exit.")
    status(repo, agents_directory)


def uninstall(repo: Path, agents_directory: Path) -> None:
    destination = agents_directory / f"{LABEL}.plist"
    if existing_plist(destination, build_plist(repo)) is None:
        print("Bot login service is not installed for this checkout.")
        return
    _bootout(f"gui/{os.getuid()}/{LABEL}")
    destination.unlink()
    print("Bot login service removed. Configuration, data, logs and models were kept.")


def compact_status(output: str) -> str:
    state_match = re.search(r"^\s*state = ([a-z ]+)\s*$", output, re.MULTILINE)
    allowed_states = {"running", "waiting", "exited", "not running", "spawn scheduled"}
    state = state_match.group(1).strip() if state_match else "loaded"
    if state not in allowed_states:
        state = "loaded"
    pid_match = re.search(r"^\s*pid = ([0-9]+)\s*$", output, re.MULTILINE)
    suffix = f" (PID {pid_match.group(1)})" if pid_match else ""
    return f"Bot login service: {state}{suffix}."


def status(repo: Path, agents_directory: Path) -> None:
    destination = agents_directory / f"{LABEL}.plist"
    if existing_plist(destination, build_plist(repo)) is None:
        print("Bot login service is not installed for this checkout.")
        return
    result = run_launchctl("print", f"gui/{os.getuid()}/{LABEL}", check=False)
    if result.returncode:
        print("Bot login service definition exists; launchd status is unavailable.")
        return
    print(compact_status(result.stdout))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "uninstall", "status"])
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This helper requires macOS.")
    repo = Path(__file__).resolve().parents[1]
    agents_directory = Path.home() / "Library" / "LaunchAgents"
    try:
        {"install": install, "uninstall": uninstall, "status": status}[args.command](
            repo, agents_directory
        )
    except (ServiceError, ConfigurationError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except (OSError, ValueError):
        print("The service operation failed. Check local file access and retry.", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
