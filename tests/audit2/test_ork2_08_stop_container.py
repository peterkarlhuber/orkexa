"""ORK2-08: stop_container returns at once for a container it killed, and still settles otherwise."""

from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

from orkexa.dynamic.docker import _container
from orkexa.dynamic.docker._container import stop_container

pytestmark = pytest.mark.skipif(os.name != "posix", reason="a shell script plays docker")

# kill exits with the code in kill.rc; ps prints ps.out once per line of ps.seq (then nothing);
# every call is logged. (No `sed -i`: BSD sed on macOS needs a suffix argument.)
_DOCKER = """#!/bin/sh
dir="$(dirname "$0")"
echo "$1" >> "$dir/log"
case "$1" in
  kill) exit "$(cat "$dir/kill.rc")" ;;
  rm) exit 0 ;;
  ps)
    if [ -s "$dir/ps.seq" ]; then
      head -n 1 "$dir/ps.seq"
      tail -n +2 "$dir/ps.seq" > "$dir/ps.rest"; mv "$dir/ps.rest" "$dir/ps.seq"
    fi
    exit 0 ;;
esac
exit 1
"""


def _docker(tmp_path: Path, *, kill_rc: int, ps: list[str]) -> str:
    script = tmp_path / "docker"
    script.write_text(_DOCKER)
    script.chmod(0o755)
    (tmp_path / "kill.rc").write_text(str(kill_rc))
    (tmp_path / "ps.seq").write_text("".join(f"{line}\n" for line in ps))
    return str(script)


def _log(tmp_path: Path) -> list[str]:
    return (tmp_path / "log").read_text().split()


def test_a_killed_container_is_confirmed_gone_without_the_settle_wait(tmp_path: Path) -> None:
    docker = _docker(tmp_path, kill_rc=0, ps=[])
    started = time.monotonic()
    stop_container(docker, "orkexa-x")
    assert time.monotonic() - started < 1.0  # before: always >= 3 s
    assert _log(tmp_path) == ["kill", "rm", "ps"]


def test_a_killed_container_still_listed_is_removed_again(tmp_path: Path) -> None:
    docker = _docker(tmp_path, kill_rc=0, ps=["abc123", ""])
    stop_container(docker, "orkexa-x")
    assert _log(tmp_path) == ["kill", "rm", "ps", "rm", "ps"]


class _Clock:
    """A fake clock for ``_container.time``: it only moves when ``sleep`` is called, so how many polls fit
    into the settle time does not depend on how fast the runner starts the fake docker script. Everything
    else is the real ``time`` module."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def __getattr__(self, name: str) -> object:
        return getattr(time, name)


def test_without_a_kill_the_name_is_watched_for_the_settle_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The client may have been killed with its create request in flight: the container can appear later.
    clock = _Clock()
    monkeypatch.setattr(_container, "time", clock)
    monkeypatch.setattr(_container, "_SETTLE_S", 3.0)  # exact in binary, so the poll count is exact too
    monkeypatch.setattr(_container, "_SETTLE_POLL_S", 0.5)
    docker = _docker(tmp_path, kill_rc=1, ps=["", "late42"])
    stop_container(docker, "orkexa-x")
    # Watched for the whole settle time, one poll interval before each check.
    assert clock.now == 3.0 and clock.sleeps == [0.5] * 6
    # Empty, then the late container (removed), then the watch goes on after the late removal until the
    # settle time is over: six checks in all.
    assert _log(tmp_path) == ["kill", "rm", "ps", "ps", "rm", "ps", "ps", "ps", "ps"]
