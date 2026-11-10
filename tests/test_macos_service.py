import importlib.util
import plistlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "macos_service.py"
SPEC = importlib.util.spec_from_file_location("macos_service", SCRIPT)
service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(service)


def test_plist_preserves_spaces_and_excludes_credentials(tmp_path: Path, monkeypatch):
    repo = tmp_path / "My Projects" / "Voice Bot"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "private-token-marker")
    monkeypatch.setenv("ALLOWED_USER_IDS", "private-user-marker")
    definition = service.build_plist(repo)
    encoded = plistlib.dumps(definition)
    restored = plistlib.loads(encoded)
    assert restored["Label"] == service.LABEL
    assert restored["ProgramArguments"] == [
        str(repo / ".venv/bin/python"),
        "-m",
        "voice_summary_bot.cli",
        "--env-file",
        str(repo / ".env"),
        "bot",
    ]
    assert restored["WorkingDirectory"] == str(repo)
    assert restored["EnvironmentVariables"] == {"PATH": service.SERVICE_PATH}
    assert restored["RunAtLoad"] is True
    assert restored["KeepAlive"] is True
    assert restored["ThrottleInterval"] == 30
    assert restored["StandardOutPath"] == str(repo / "data/bot-service.log")
    assert b"private-token-marker" not in encoded
    assert b"private-user-marker" not in encoded


def test_another_checkout_is_rejected_before_service_changes(tmp_path: Path, monkeypatch):
    repo = tmp_path / "ours"
    agents = tmp_path / "agents"
    agents.mkdir()
    definition = agents / f"{service.LABEL}.plist"
    definition.write_bytes(plistlib.dumps(service.build_plist(tmp_path / "other")))
    calls = []
    monkeypatch.setattr(service, "run_launchctl", lambda *a, **k: calls.append(a))
    with pytest.raises(service.ServiceError, match="different checkout"):
        service.install(repo, agents)
    with pytest.raises(service.ServiceError, match="different checkout"):
        service.uninstall(repo, agents)
    assert calls == []
    assert definition.exists()


def test_install_and_restart_owned_definition_use_private_files(tmp_path: Path, monkeypatch):
    repo = tmp_path / "our repo"
    repo.mkdir()
    agents = tmp_path / "LaunchAgents"
    monkeypatch.setattr(service, "validate_install", lambda _: None)
    calls = []

    def launchctl(*arguments, **kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "state = running\npid = 42\n", "")

    monkeypatch.setattr(service, "run_launchctl", launchctl)
    service.install(repo, agents)
    definition = agents / f"{service.LABEL}.plist"
    assert definition.stat().st_mode & 0o777 == 0o600
    assert (repo / "data").stat().st_mode & 0o777 == 0o700
    assert (repo / "data/bot-service.log").stat().st_mode & 0o777 == 0o600
    assert [call[0] for call in calls] == ["enable", "bootstrap", "print"]
    calls.clear()
    service.install(repo, agents)
    assert [call[0] for call in calls] == ["print", "enable", "kickstart", "print"]
    assert calls[2][1] == "-k"
    assert definition.stat().st_mode & 0o777 == 0o600
    (repo / "data/saved-marker").write_text("keep")
    service.uninstall(repo, agents)
    assert not definition.exists()
    assert (repo / "data/saved-marker").read_text() == "keep"


def test_changed_definition_retries_transient_bootstrap_without_repeated_bootout(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "repo"
    agents = tmp_path / "agents"
    agents.mkdir()
    definition = agents / f"{service.LABEL}.plist"
    previous = service.build_plist(repo)
    previous["ThrottleInterval"] = 60
    definition.write_bytes(plistlib.dumps(previous))
    monkeypatch.setattr(service, "validate_install", lambda _: None)
    calls, delays = [], []
    bootstrap_results = iter([5, 0])

    def launchctl(*arguments, **kwargs):
        calls.append(arguments)
        code = next(bootstrap_results) if arguments[0] == "bootstrap" else 0
        return subprocess.CompletedProcess(arguments, code, "state = running\n", "")

    monkeypatch.setattr(service, "run_launchctl", launchctl)
    monkeypatch.setattr(service.time, "sleep", delays.append)
    service.install(repo, agents)
    assert [call[0] for call in calls] == [
        "bootout",
        "enable",
        "bootstrap",
        "bootstrap",
        "print",
    ]
    assert calls[2] == calls[3]
    assert delays == [5]
    assert plistlib.loads(definition.read_bytes()) == service.build_plist(repo)


def test_same_definition_not_loaded_uses_bootstrap_fallback(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    agents = tmp_path / "agents"
    agents.mkdir()
    definition = agents / f"{service.LABEL}.plist"
    definition.write_bytes(plistlib.dumps(service.build_plist(repo)))
    monkeypatch.setattr(service, "validate_install", lambda _: None)
    calls = []

    def launchctl(*arguments, **kwargs):
        calls.append(arguments)
        code = 3 if len(calls) == 1 or arguments[0] == "bootout" else 0
        return subprocess.CompletedProcess(arguments, code, "state = running\n", "")

    monkeypatch.setattr(service, "run_launchctl", launchctl)
    service.install(repo, agents)
    assert [call[0] for call in calls] == [
        "print",
        "bootout",
        "enable",
        "bootstrap",
        "print",
    ]


@pytest.mark.parametrize("codes", [[5, 5, 5, 5], [7], [5, 7]])
def test_bootstrap_stops_after_bounded_or_nonretryable_errors(tmp_path, monkeypatch, codes):
    calls, delays = [], []
    results = iter(codes)

    def launchctl(*arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(
            arguments, next(results), "private-output-marker", "private-error-marker"
        )

    monkeypatch.setattr(service, "run_launchctl", launchctl)
    monkeypatch.setattr(service.time, "sleep", delays.append)
    destination = tmp_path / "service.plist"
    with pytest.raises(service.ServiceError, match=f"exit {codes[-1]}") as error:
        service._bootstrap("gui/123", destination, reinstall=True)
    assert "private" not in str(error.value)
    assert calls == [(("bootstrap", "gui/123", str(destination)), {"check": False})] * len(codes)
    assert delays == [5] * (len(codes) - 1)


def test_first_install_does_not_retry_bootstrap_error(tmp_path, monkeypatch):
    calls = []

    def launchctl(*arguments, **kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 5, "", "")

    monkeypatch.setattr(service, "run_launchctl", launchctl)
    monkeypatch.setattr(service.time, "sleep", lambda _: pytest.fail("Unexpected retry"))
    with pytest.raises(service.ServiceError, match="exit 5"):
        service._bootstrap("gui/123", tmp_path / "service.plist", reinstall=False)
    assert len(calls) == 1


def test_launchctl_never_inherits_private_environment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "private-token-marker")
    seen = {}

    def run(arguments, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(service.subprocess, "run", run)
    service.run_launchctl("print", "gui/123/example")
    assert seen["env"] == {"PATH": service.SERVICE_PATH}


@pytest.mark.parametrize("model_ready", [False, True])
def test_validation_uses_checkout_env_and_restores_shell_environment(
    tmp_path: Path, monkeypatch, model_ready
):
    repo = tmp_path / "our repo"
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch(mode=0o700)
    model = repo / "model.bin"
    model.touch()
    (repo / ".env").write_text("# synthetic fixture only\n")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "shell-marker")
    original_directory = Path.cwd()

    def load(path, *, require_bot):
        assert path == repo / ".env"
        assert require_bot
        assert Path.cwd() == repo
        assert "TELEGRAM_BOT_TOKEN" not in service.os.environ
        service.os.environ["TELEGRAM_BOT_TOKEN"] = "file-marker"
        return SimpleNamespace(
            model_path=model,
            whisper_binary="whisper-cli",
            ffmpeg_binary="ffmpeg",
            vad_model_path=model,
            vad_binary="whisper-vad-speech-segments",
        )

    monkeypatch.setattr(service.Settings, "load", load)

    def verify(path):
        from voice_summary_bot.errors import SpeechCheckUnavailableError

        assert path == model
        if not model_ready:
            raise SpeechCheckUnavailableError()

    monkeypatch.setattr("voice_summary_bot.speech_presence.verify_vad_model", verify)
    monkeypatch.setattr(service.shutil, "which", lambda command, *, path: f"/bin/{command}")
    if model_ready:
        service.validate_install(repo)
    else:
        with pytest.raises(service.ServiceError, match="speech detector model"):
            service.validate_install(repo)
    assert Path.cwd() == original_directory
    assert service.os.environ["TELEGRAM_BOT_TOKEN"] == "shell-marker"


def test_status_never_prints_raw_launchctl_environment():
    output = """service = {
        state = running
        environment = { TELEGRAM_BOT_TOKEN = private-token-marker }
        pid = 123
    }"""
    assert service.compact_status(output) == "Bot login service: running (PID 123)."


def test_failed_bootout_does_not_remove_definition(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    agents = tmp_path / "agents"
    agents.mkdir()
    definition = agents / f"{service.LABEL}.plist"
    definition.write_bytes(plistlib.dumps(service.build_plist(repo)))
    monkeypatch.setattr(
        service,
        "run_launchctl",
        lambda *a, **k: subprocess.CompletedProcess(a, 5, "", "private-error-marker"),
    )
    with pytest.raises(service.ServiceError, match="could not be stopped"):
        service.uninstall(repo, agents)
    assert definition.exists()
