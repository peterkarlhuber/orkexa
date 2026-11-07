"""Test helper: a fake ``docker`` executable (``tests/fake_docker.py``) with its state in a temp dir."""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from orkexa.dynamic import DockerSandbox
from orkexa.dynamic.docker import SELF_TEST_SCRIPT

HOSTS: dict[str, dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "docker" / "hosts.json").read_text(encoding="utf-8")
)
FAKE = Path(__file__).with_name("fake_docker.py")
IMAGE_ID = "sha256:" + "ab" * 32
PYTHON_ENV = [
    "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305",
    "PYTHON_VERSION=3.12.7",
    "PYTHON_SHA256=24887b92e2afd4a2ac602419ad4b596372f67ac9b077190f459aba390faf5550",
    "LANG=C.UTF-8",
]

needs_posix = pytest.mark.skipif(os.name != "posix", reason="the fake docker CLI is a POSIX script")


class FakeDocker:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "docker"
        self.path.write_text(
            f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(FAKE)!r}, run_name='__main__')\n"
        )
        self.path.chmod(0o755)
        self.host("rootful-cgroup2")
        self.image("orkexa-sandbox")

    def write(self, name: str, data: Any) -> None:
        (self.directory / name).write_text(json.dumps(data))

    def read(self, name: str, default: Any) -> Any:
        path = self.directory / name
        return json.loads(path.read_text()) if path.exists() else default

    def host(self, name: str, **info: Any) -> None:
        host = json.loads(json.dumps(HOSTS[name]))
        host["info"].update(info)
        self.write("host.json", host)

    def image(
        self,
        ref: str,
        *,
        image_id: str = IMAGE_ID,
        env: list[str] | None = None,
        digests: list[str] | None = None,
    ) -> None:
        images = self.read("images.json", {})
        images[ref] = {
            "Id": image_id,
            "RepoTags": [ref],
            "RepoDigests": digests or [],
            "Config": {"Env": PYTHON_ENV if env is None else env},
        }
        self.write("images.json", images)

    def self_test(self, **changes: Any) -> None:
        self.write("selftest.json", changes)

    def fail(self, command: str, code: int = 1) -> None:
        self.write("fail.json", {**self.read("fail.json", {}), command: code})

    def calls(self) -> list[list[str]]:
        path = self.directory / "log.jsonl"
        if not path.exists():
            return []
        text = path.read_text()
        # A running ``docker`` may be appending a line right now; read only the lines it finished.
        return [json.loads(line) for line in text[: text.rfind("\n") + 1].splitlines()]

    def runs(self) -> list[list[str]]:
        """The ``docker run`` calls of tool code (self-tests left out)."""
        return [call for call in self.calls() if call[:1] == ["run"] and SELF_TEST_SCRIPT not in call]

    def self_tests(self) -> list[list[str]]:
        return [call for call in self.calls() if call[:1] == ["run"] and SELF_TEST_SCRIPT in call]

    def containers(self) -> list[dict[str, Any]]:
        return list(self.read("containers.json", []))


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDocker]:
    fake = FakeDocker(tmp_path / "fake-docker")
    monkeypatch.setenv("FAKE_DOCKER_DIR", str(fake.directory))
    DockerSandbox.clear_self_test_cache()
    yield fake
    DockerSandbox.clear_self_test_cache()
