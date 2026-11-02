"""``python -m orkexa.dynamic.docker``: check this computer's Docker sandbox (manual and CI verification)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import sys

from ..capabilities import SandboxProbe
from ..errors import SandboxError
from ._container import reap
from ._host import DockerProbeError, inspect_host, preflight


def _print(probe: SandboxProbe, as_json: bool) -> int:
    if as_json:
        print(json.dumps(probe.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"{probe.backend}: {'available' if probe.available else 'refused'}")
        if not probe.available:
            print(f"reason: {probe.reason}")
            print(f"detail: {probe.detail}")
        for key, value in probe.facts:
            print(f"  {key}: {value}")
    return 0 if probe.available else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code (0: the sandbox can run here, 1: refused)."""
    from ..sandbox import DockerSandbox  # the sandbox module imports this package

    parser = argparse.ArgumentParser(
        prog="python -m orkexa.dynamic.docker",
        description="Check whether DockerSandbox can run here (preflight and in-container self-test).",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--self-test", action="store_true", help="preflight and self-test (the default)")
    action.add_argument("--preflight", action="store_true", help="only inspect the daemon")
    action.add_argument("--reap", action="store_true", help="remove containers past their deadline")
    parser.add_argument("--docker", default="docker", help="the docker CLI (default: docker)")
    parser.add_argument(
        "--image", default="orkexa-sandbox", help="the sandbox image (default: orkexa-sandbox)"
    )
    parser.add_argument("--owner", default="orkexa", help="the owner label (default: orkexa)")
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--cpus", type=float, default=1.0)
    parser.add_argument("--pids", type=int, default=64)
    parser.add_argument(
        "--remove-all", action="store_true", help="with --reap: also containers before deadline"
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    options = parser.parse_args(argv)

    if options.reap:
        try:
            removed = reap(options.docker, options.owner, remove_all=options.remove_all)
        except (SandboxError, ValueError) as exc:
            print(f"reap failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"removed": removed}) if options.json else f"removed {removed} container(s)")
        return 0
    if options.preflight:
        try:
            probe = preflight(inspect_host(options.docker))
        except DockerProbeError as exc:
            probe = SandboxProbe("docker", False, exc.reason, exc.detail)
        return _print(probe, options.json)
    sandbox = DockerSandbox(
        docker=options.docker,
        image=options.image,
        owner=options.owner,
        memory_mb=options.memory_mb,
        cpus=options.cpus,
        pids_limit=options.pids,
    )
    return _print(sandbox.self_test(), options.json)
