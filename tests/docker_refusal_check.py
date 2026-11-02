"""CI check (``docker`` job, mode ``refused``): a rootless daemon without cpu delegation is refused.

Run as ``python tests/docker_refusal_check.py`` against such a daemon; exits non-zero unless the preflight,
the probe and a real call all refuse it with the reason ``limits`` and no code runs.
"""

from __future__ import annotations

import json

from orkexa import ToolExecutionError
from orkexa.dynamic import DockerSandbox, SandboxError
from orkexa.dynamic.docker import inspect_host, preflight

CODE = "def one() -> int:\n    '''One.\n\n    Returns:\n        One.\n    '''\n    return 1\n"


def main() -> None:
    host = inspect_host()
    print(host)
    assert host.rootless, "the job must run a rootless daemon"
    checked = preflight(host)
    probe = DockerSandbox.probe()
    print(json.dumps(probe.as_dict(), indent=2))
    assert not checked.available and checked.reason == "limits", checked
    assert not probe.available and probe.reason == "limits", probe
    assert not DockerSandbox.available()
    try:
        DockerSandbox().call(CODE, "one", {})
    except ToolExecutionError as exc:
        assert isinstance(exc.__cause__, SandboxError) and "(limits)" in str(exc), exc
    else:
        raise AssertionError("the sandbox ran code on a refused daemon")
    print("refused as required")


if __name__ == "__main__":
    main()
