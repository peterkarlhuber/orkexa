"""Every example runs end to end offline: the SDK client classes are replaced by the
scripted clients of the contract harnesses, so the examples exercise the real
adapters, runtime and public API without network access or API keys."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from typing import Any

import anthropic
from google import genai
import openai
import pytest

from orkexa.dynamic import DockerSandbox

from .anthropic.dynamic_helpers import MARGIN
from .contract.harness import AnthropicHarness, ChatHarness, GeminiHarness, OpenAIHarness, Script

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

pytestmark = [
    pytest.mark.filterwarnings("ignore::orkexa.HistoryConversionWarning"),
    pytest.mark.filterwarnings("ignore::orkexa.dynamic.UnpricedUsageWarning"),
]


class Offline:
    def __init__(self) -> None:
        self.anthropic = AnthropicHarness()
        self.openai = OpenAIHarness()
        self.gemini = GeminiHarness()
        self.chat = ChatHarness()

    def script(self, **scripts: list[Any]) -> None:
        for name, responses in scripts.items():
            getattr(self, name).script = Script(responses)

    def openai_client(self, asynchronous: bool) -> Any:
        responses = self.openai.client(asynchronous).responses
        chat = self.chat.client(asynchronous).chat
        return SimpleNamespace(responses=responses, chat=chat)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[Offline]:
    world = Offline()
    for variable in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(variable, "offline-test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ORKEXA_MODEL", raising=False)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: world.anthropic.client(False))
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **kw: world.anthropic.client(True))
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: world.openai_client(False))
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: world.openai_client(True))
    monkeypatch.setattr(genai, "Client", lambda **kw: world.gemini.client(False))
    monkeypatch.syspath_prepend(str(EXAMPLES))
    sys.modules.pop("_setup", None)
    yield world
    sys.modules.pop("_setup", None)


def run(name: str) -> None:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")


def test_every_example_is_covered() -> None:
    scripts = sorted(p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py"))
    assert len(scripts) == 13
    tested = {name.split("_")[2] for name in globals() if name.startswith("test_example_")}
    assert tested == {s[:2] for s in scripts}


def test_example_01(offline: Offline, capsys: pytest.CaptureFixture[str]) -> None:
    offline.script(
        anthropic=[offline.anthropic.text("A Claude answer.")],
        openai=[offline.openai.text("A GPT answer.")],
        gemini=[offline.gemini.text("A Gemini answer.")],
    )
    run("01_text_agents.py")
    out = capsys.readouterr().out
    assert "A Claude answer." in out and "A GPT answer." in out and "A Gemini answer." in out


def test_example_02(offline: Offline, capsys: pytest.CaptureFixture[str]) -> None:
    call = [("c1", "convert_currency", {"amount": 250, "rate": 1.08})]
    offline.script(
        anthropic=[offline.anthropic.tool_calls(call), offline.anthropic.text("270 USD")],
        openai=[offline.openai.tool_calls(call), offline.openai.text("270 USD")],
        gemini=[offline.gemini.tool_calls(call), offline.gemini.text("270 USD")],
    )
    run("02_same_tool_three_providers.py")
    assert capsys.readouterr().out.count("270 USD") == 3
    for harness in (offline.anthropic, offline.openai, offline.gemini):
        assert harness.tool_results(harness.script.calls[1])[0][1] == "270.0"


def test_example_03(offline: Offline, capsys: pytest.CaptureFixture[str]) -> None:
    offline.script(
        chat=[
            offline.chat.tool_calls([("w1", "word_count", {"text": "the quick brown fox"})]),
            offline.chat.text("4 words"),
        ]
    )
    run("03_compatible_endpoint.py")
    out = capsys.readouterr().out
    assert "supported unknown" in out and "4 words" in out
    assert offline.chat.tool_results(offline.chat.script.calls[1])[0][1] == "4"


def test_example_04(
    offline: Offline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ORKEXA_MODEL", "openai:gpt-5")
    o = offline.openai
    offline.script(
        openai=[
            o.tool_calls([("p1", "price", {"item": "apple"}), ("p2", "price", {"item": "kiwi"})]),
            o.tool_calls([("p3", "price", {"item": "pear"}), ("m1", "multiply", {"a": 7, "b": 0.5})]),
            o.tool_calls([("m2", "multiply", {"a": 3, "b": 0.8})]),
            o.text("EUR 5.90"),
            o.tool_calls([("s1", "stock", {"item": n}) for n in ("apple", "pear", "melon")]),
            o.text("40, 12 and 3"),
        ]
    )
    run("04_tool_loop_sync_async.py")
    out = capsys.readouterr().out
    assert "EUR 5.90" in out and "40, 12 and 3" in out
    first_round = o.tool_results(o.script.calls[1])
    assert first_round[1][2] is True  # unknown item went back to the model as an error
    assert [r[1] for r in o.tool_results(o.script.calls[5])] == ["40", "12", "3"]


def test_example_05(
    offline: Offline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ORKEXA_MODEL", "openai:gpt-5")
    # strict mode: every field is sent, optional ones as null -> restored to their defaults
    payload = '{"name": "Voestalpine", "country": "AT", "founded": 1938, "products": null}'
    offline.script(openai=[offline.openai.text(payload)])
    run("05_structured_output.py")
    out = capsys.readouterr().out
    assert '"founded": 1938' in out and '"products": []' in out
    text_format = offline.openai.script.calls[0]["text"]["format"]
    assert text_format["strict"] is True and text_format["schema"]["required"] == [
        "name",
        "country",
        "founded",
        "products",
    ]


def test_example_06(
    offline: Offline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ORKEXA_MODEL", "gemini:gemini-3-pro")
    g = offline.gemini
    offline.script(
        gemini=[
            g.tool_calls([("d1", "delete_everything", {"confirm": True})]),
            g.text("Rivers run, I refused."),
            g.text("one two three four five"),
            g.truncated("Europe began"),
        ]
    )
    run("06_streaming_hooks_errors.py")
    out = capsys.readouterr().out
    assert "Rivers run" in out and "[usage] gemini" in out and "[turn completed]" in out
    assert "text deltas: 'one two three four five'" in out
    assert "truncated:" in out
    denied = g.tool_results(g.script.calls[1])[0]
    assert denied[2] is True and "Denied by policy" in denied[1]


def test_example_07(
    offline: Offline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ORKEXA_MODEL", "gemini:gemini-3-pro")
    offline.script(gemini=[offline.gemini.text("400")])
    run("07_history_migration.py")
    out = capsys.readouterr().out
    assert "strict: The conversation contains reasoning content produced by anthropic" in out
    assert "saved: orkexa.history version 1" in out and "-> 400" in out
    assert offline.gemini.user_texts(offline.gemini.script.calls[0])[0] == "What is 17 * 23?"


def test_example_08(
    offline: Offline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ORKEXA_MODEL", "openai:gpt-5")
    o = offline.openai
    offline.script(
        openai=[
            o.tool_calls([("s1", "search", {"query": "warranty", "k": 3})]),
            o.text("24 months"),
            o.tool_calls(
                [("m1", "memory", {"command": "create", "path": "/memories/q3.md", "file_text": "EUR 4.2m"})]
            ),
            o.text("Saved."),
            o.text("Black."),
        ]
    )
    run("08_memory_files_rag.py")
    out = capsys.readouterr().out
    assert "verified source: handbook.md / Warranty" in out
    assert "Saved." in out and "Black." in out
    image_prompt = o.script.calls[4]["input"][-1]["content"]
    assert image_prompt[0]["type"] == "input_image"


def test_example_09(offline: Offline, capsys: pytest.CaptureFixture[str]) -> None:
    offline.script(
        anthropic=[offline.anthropic.text("Danube, Inn, Mur")],
        openai=[offline.openai.text("Danube, Inn, Mur")],
        gemini=[offline.gemini.text("Danube, Inn, Mur")],
    )
    run("09_usage_budget.py")
    out = capsys.readouterr().out
    for provider in ("anthropic", "openai", "gemini"):
        assert f"{provider}: in=80 out=50 cached=20 reasoning=10" in out
    assert "refused" not in out and "pricing=registered" in out


def test_example_10(offline: Offline, capsys: pytest.CaptureFixture[str]) -> None:
    verdict = (
        '{"criteria": [{"name": "sourced", "reasoning": "ok", "passed": true}], '
        '"summary": "good", "suggestions": []}'
    )
    a = offline.anthropic
    offline.script(
        anthropic=[
            a.tool_calls(
                [("t1", "research", {"question": "Danube trade?"}), ("t2", "review", {"draft": "d"})]
            ),
            a.text("The Danube carries freight (source: researcher)."),
        ],
        openai=[offline.openai.text("Freight and tourism.")],
        gemini=[offline.gemini.text("- cite sources"), offline.gemini.text(verdict)],
    )
    run("10_mixed_provider_delegation.py")
    out = capsys.readouterr().out
    assert "judge passed: True after 1 answer(s)" in out
    assert {line.split(":")[0] for line in out.splitlines() if line.endswith("tokens")} == {
        "anthropic",
        "openai",
        "gemini",
    }


def test_example_11(
    offline: Offline, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ORKEXA_MODEL", "openai:gpt-5")
    monkeypatch.setattr(DockerSandbox, "available", staticmethod(lambda docker="docker": False))
    monkeypatch.setenv("ORKEXA_ALLOW_SUBPROCESS_SANDBOX", "1")
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    o = offline.openai
    offline.script(
        openai=[
            o.tool_calls([("c1", "create_tool", {"requirement": "operating margin"})]),
            o.text(MARGIN.model_dump_json()),  # the Coding Agent's proposal
            o.tool_calls(
                [("c2", "calculate_operating_margin", {"revenue": 200.0, "operating_income": 50.0})]
            ),
            o.text("The operating margin is 25%."),
        ]
    )
    run("11_dynamic_tools_sandbox.py")
    out = capsys.readouterr().out
    assert "approval requested: tool calculate_operating_margin" in out
    assert "The operating margin is 25%." in out
    assert o.tool_results(o.script.calls[3])[0][1] == "0.25"


def test_example_11_stops_without_an_isolated_sandbox(
    offline: Offline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORKEXA_MODEL", "openai:gpt-5")
    monkeypatch.setattr(DockerSandbox, "available", staticmethod(lambda docker="docker": False))
    monkeypatch.delenv("ORKEXA_ALLOW_SUBPROCESS_SANDBOX", raising=False)
    with pytest.raises(SystemExit, match="not a security boundary"):
        run("11_dynamic_tools_sandbox.py")
    assert offline.openai.script.calls == []


def test_example_12(capsys: pytest.CaptureFixture[str]) -> None:
    run("12_pricing_snapshots.py")  # offline by design: no client is patched
    out = capsys.readouterr().out
    assert "total $0.160000 from snapshot acme-2026-11 (acme-contract, contract)" in out
    assert "priority tier: total=None missing=['tariff:service_tier=priority']" in out
    assert "pinned acme-2026-11: $1.25, current acme-2026-12: $1.00" in out
    assert "record: cost=$0.00002 version=acme-2026-12 quality=estimated source=acme-contract" in out


def test_example_13(capsys: pytest.CaptureFixture[str]) -> None:
    run("13_flow_quality_loop.py")  # offline by design: the agents use the fake provider
    out = capsys.readouterr().out
    assert "round 1: AGAIN: name the providers" in out
    assert "suspended: waiting for ['approve//1']" in out
    assert "succeeded: drafts ['draft/0/1', 'draft/1/1'], loop counters {'retry@': 1}" in out
    assert "published: Orkexa runs agents on four providers with one API." in out
