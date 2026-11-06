"""Messages for developers: what failed and what to do next (ORK-037). No network."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket

import pytest

from orkexa import Agent, ConfigurationError
from orkexa.local import ManagedOllama, OllamaClient, OllamaError
from orkexa.local import __main__ as cli
from orkexa.local.convert import MDENSEON, build_model
from orkexa.local.install import install_runtime
from orkexa.providers import resolve_model


@pytest.mark.parametrize(
    ("model", "example"),
    [
        ("anthropic:", "'anthropic:claude-sonnet-5-5'"),
        ("openai: ", "'openai:gpt-5.5'"),
        ("ollama:", "'ollama:qwen3.5:4b'"),
        ("compatible:", "'compatible:my-model'"),
    ],
)
def test_provider_prefix_without_a_model_is_refused_with_an_example(model: str, example: str) -> None:
    with pytest.raises(ConfigurationError, match="no model") as caught:
        resolve_model(model)
    assert example in str(caught.value)
    with pytest.raises(ConfigurationError, match="no model"):
        Agent(model=model)


def test_prefixed_models_still_resolve() -> None:
    assert resolve_model("ollama:qwen3.5:4b") == ("ollama", "qwen3.5:4b")


def _every_action(parser: argparse.ArgumentParser) -> list[tuple[str, argparse.Action]]:
    found: list[tuple[str, argparse.Action]] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                found.extend((f"{name} {dest}", item) for dest, item in _every_action(sub))
        elif not isinstance(action, argparse._HelpAction):
            found.append((action.dest, action))
    return found


def test_every_cli_option_and_command_has_a_help_text() -> None:
    parser = cli.parser()
    missing = [name for name, action in _every_action(parser) if not action.help]
    assert missing == []
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    helped = {choice.dest for choice in commands._choices_actions if choice.help}
    assert helped == set(commands.choices)


def test_cli_reports_any_orkexa_error_in_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(args: argparse.Namespace) -> int:
        raise ConfigurationError("something is configured wrong")

    monkeypatch.setattr(cli, "_cmd_scan", fail)
    assert cli.main(["--home", str(tmp_path), "scan", str(tmp_path)]) == 1
    assert capsys.readouterr().err == "error (ConfigurationError): something is configured wrong\n"


def test_cli_reports_a_missing_runtime_with_the_next_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--home", str(tmp_path), "serve"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error (runtime_missing): ")
    assert "orkexa-local install-runtime --download" in err


def test_cli_reports_os_errors_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--home", str(tmp_path), "verify", str(tmp_path / "missing")]) == 1
    assert capsys.readouterr().err.startswith("error (")


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_status_names_the_next_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "detect_ollama", lambda: None)
    url = f"http://127.0.0.1:{_closed_port()}"
    assert cli.main(["--home", str(tmp_path), "--url", url, "status"]) == 0
    out = capsys.readouterr().out
    assert "next: orkexa-local install-runtime --download" in out
    assert "orkexa-local serve" in out  # in the unreachable message


def test_unreachable_server_names_how_to_start_it() -> None:
    client = OllamaClient(f"http://127.0.0.1:{_closed_port()}", timeout=2)
    with pytest.raises(OllamaError) as caught:
        client.version()
    assert caught.value.code == "unreachable"
    assert "orkexa-local serve" in str(caught.value)


def test_runtime_and_server_messages_name_the_next_step(tmp_path: Path) -> None:
    with pytest.raises(OllamaError, match="install-runtime --archive FILE"):
        install_runtime(tmp_path)
    server = ManagedOllama(tmp_path)
    with pytest.raises(OllamaError, match="install-runtime --download"):
        server.start(timeout=1)
    with pytest.raises(OllamaError, match="orkexa-local serve"):
        server.client()


def test_build_without_inputs_names_the_options(tmp_path: Path) -> None:
    client = OllamaClient(f"http://127.0.0.1:{_closed_port()}", timeout=1)
    with pytest.raises(OllamaError) as caught:
        build_model(client, MDENSEON, work_dir=tmp_path)
    assert caught.value.code == "source_missing"
    assert "--weights FILE --template FILE" in str(caught.value)
    assert "--download" in str(caught.value)
    with pytest.raises(OllamaError, match="check the --weights/--template path"):
        build_model(
            client,
            MDENSEON,
            weights=tmp_path / "nope.safetensors",
            template=tmp_path / "nope.gguf",
            work_dir=tmp_path,
        )
