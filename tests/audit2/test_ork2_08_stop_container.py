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


def test_without_a_kill_the_name_is_watched_for_the_settle_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The client may have been killed with its create request in flight: the container can appear later.
    monkeypatch.setattr(_container, "_SETTLE_S", 0.6)
    monkeypatch.setattr(_container, "_SETTLE_POLL_S", 0.1)
    docker = _docker(tmp_path, kill_rc=1, ps=["", "late42"])
    started = time.monotonic()
    stop_container(docker, "orkexa-x")
    assert time.monotonic() - started >= 0.6
    log = _log(tmp_path)
    assert log[:2] == ["kill", "rm"] and log.count("rm") == 2  # the late container was removed
    # Empty, then the late container, then at least one more check: watching goes on after the late
    # removal. Only a lower bound on polls, since a slow runner (macOS) fits fewer into the settle time.
    assert log.count("ps") >= 3
