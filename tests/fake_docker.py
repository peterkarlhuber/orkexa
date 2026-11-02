"""A stand-in for the ``docker`` CLI in the unit tests (no daemon needed).

The state lives in the directory ``$FAKE_DOCKER_DIR``: ``host.json`` (``{"info": ..., "version": ...}``,
the outputs of ``docker info``/``docker version``), ``images.json`` and ``pullable.json`` (``docker image
inspect`` objects by reference), ``containers.json`` (what ``docker ps``/``container inspect`` show),
``selftest.json`` (changes to the self-test report) and ``log.jsonl`` (every call). ``docker run`` checks its
options strictly and runs the container's ``python -I -B -c <script>`` with this interpreter in a temporary
directory (so the stdin bootstrap really runs); the self-test script gets a report derived from the run
options instead, as a correct engine would produce it. Real containers: ``tests/test_docker_sandbox.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from typing import Any
import uuid

_VALUE_OPTIONS = {
    "--pull",
    "--name",
    "--network",
    "--cap-drop",
    "--security-opt",
    "--user",
    "--ipc",
    "--shm-size",
    "--ulimit",
    "--log-driver",
    "--pids-limit",
    "--memory",
    "--memory-swap",
    "--cpus",
    "--tmpfs",
    "-w",
    "--label",
    "-v",
}
_FLAG_OPTIONS = {"--rm", "-i", "--read-only", "-d"}


def _state() -> Path:
    return Path(os.environ["FAKE_DOCKER_DIR"])


def _load(name: str, default: Any) -> Any:
    path = _state() / name
    return json.loads(path.read_text()) if path.exists() else default


def _save(name: str, data: Any) -> None:
    (_state() / name).write_text(json.dumps(data))


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    with contextlib.suppress(OSError):
        status = Path(f"/proc/{pid}/stat").read_text().split()
        if len(status) > 2 and status[2] == "Z":
            return False
    return True


def _containers() -> list[dict[str, Any]]:
    """The containers that still exist: ``--rm`` containers vanish when their process ends."""
    kept = [
        c for c in _load("containers.json", []) if not (c.get("Rm") and c.get("Pid") and not _alive(c["Pid"]))
    ]
    return kept


def _find(ref: str) -> dict[str, Any] | None:
    for container in _containers():
        if ref in (container["Id"], container["Id"][:12], container["Name"]):
            return container
    return None


def _image(ref: str) -> dict[str, Any] | None:
    images = _load("images.json", {})
    if ref in images:
        return dict(images[ref])
    for image in images.values():
        if image.get("Id") == ref:
            return dict(image)
    return None


def _parse_run(argv: list[str]) -> tuple[dict[str, list[str]], set[str], str, list[str]]:
    values: dict[str, list[str]] = {}
    flags: set[str] = set()
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in _VALUE_OPTIONS:
            values.setdefault(arg, []).append(argv[index + 1])
            index += 2
        elif arg in _FLAG_OPTIONS:
            flags.add(arg)
            index += 1
        elif arg.startswith("-"):
            sys.stderr.write(f"unknown flag: {arg}\n")
            sys.exit(125)
        else:
            return values, flags, arg, argv[index + 1 :]
    sys.stderr.write("no image\n")
    sys.exit(125)


def _mib(text: str) -> int:
    assert text.endswith("m"), text
    return int(text[:-1]) * 1024 * 1024


def _tmpfs(values: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    mounts = {}
    for spec in values.get("--tmpfs", []):
        point, _, options = spec.partition(":")
        mounts[point] = dict(
            (part.split("=", 1) if "=" in part else (part, "")) for part in options.split(",") if part
        )
    return mounts


def _merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    for key, value in changes.items():
        if value is None:
            base.pop(key, None)
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _self_test_report(values: dict[str, list[str]], flags: set[str], image: dict[str, Any]) -> dict[str, Any]:
    info = _load("host.json", {}).get("info") or {}
    security = " ".join(values.get("--security-opt", []))
    seccomp = "0" if "profile=unconfined" in json.dumps(info.get("SecurityOptions")) else "2"
    tmpfs = _tmpfs(values)
    user = values.get("--user", ["0:0"])[0]
    uid, gid = (int(part) for part in user.split(":"))
    memory = _mib(values["--memory"][0]) if "--memory" in values else None
    cpus = float(values["--cpus"][0]) if "--cpus" in values else None
    ulimits = dict(item.split("=", 1) for item in values.get("--ulimit", []))

    def rlimit(name: str) -> list[int]:
        if name not in ulimits:
            return [-1, -1]
        soft, _, hard = ulimits[name].partition(":")
        return [int(soft), int(hard or soft)]

    def mount(point: str) -> list[str] | None:
        if point not in tmpfs:
            return None
        options = {key for key in tmpfs[point] if key in ("rw", "noexec", "nosuid", "nodev")}
        return sorted(options | {"relatime"})

    env = [entry.split("=", 1)[0] for entry in (image.get("Config") or {}).get("Env") or []]
    cgroup_version = str(info.get("CgroupVersion") or "2")
    report: dict[str, Any] = {
        "version": 1,
        "status": {
            "Uid": f"{uid}\t{uid}\t{uid}\t{uid}",
            "Gid": f"{gid}\t{gid}\t{gid}\t{gid}",
            "Groups": "",
            "CapInh": "0000000000000000",
            "CapPrm": "0000000000000000" if "ALL" in values.get("--cap-drop", []) else "00000000a80425fb",
            "CapEff": "0000000000000000" if "ALL" in values.get("--cap-drop", []) else "00000000a80425fb",
            "CapBnd": "0000000000000000" if "ALL" in values.get("--cap-drop", []) else "00000000a80425fb",
            "CapAmb": "0000000000000000",
            "NoNewPrivs": "1" if "no-new-privileges" in security else "0",
            "Seccomp": seccomp,
        },
        "uid": uid,
        "gid": gid,
        "interfaces": ["lo"] if values.get("--network") == ["none"] else ["eth0", "lo"],
        "write_root": "OSError:30" if "--read-only" in flags else "allowed",
        "mounts": {
            "/": ["relatime", "ro" if "--read-only" in flags else "rw"],
            "/tmp": mount("/tmp"),
            "/work": mount("/work"),
            "/dev/shm": ["nodev", "noexec", "nosuid", "relatime", "rw"],
        },
        "sizes": {
            "/tmp": _mib(tmpfs["/tmp"]["size"]) if "/tmp" in tmpfs else None,
            "/work": _mib(tmpfs["/work"]["size"]) if "/work" in tmpfs else None,
            "/dev/shm": _mib(values["--shm-size"][0]) if "--shm-size" in values else 64 * 1024 * 1024,
        },
        "exec_work": "PermissionError:13" if "noexec" in tmpfs.get("/work", {}) else "allowed",
        "exec_tmp": "PermissionError:13" if "noexec" in tmpfs.get("/tmp", {}) else "allowed",
        "cgroup": {"version": cgroup_version},
        "rlimits": {
            "RLIMIT_NOFILE": rlimit("nofile"),
            "RLIMIT_FSIZE": rlimit("fsize"),
            "RLIMIT_CORE": rlimit("core"),
        },
        "env": sorted({*env, "HOME", "HOSTNAME"}),
    }
    if cgroup_version == "2":
        report["cgroup"].update(
            {
                "memory.max": str(memory) if memory else "max",
                "memory.swap.max": "0"
                if memory and values.get("--memory-swap") == values.get("--memory")
                else "max",
                "pids.max": values.get("--pids-limit", ["max"])[0],
                "cpu.max": f"{int(cpus * 100000)} 100000" if cpus else "max 100000",
            }
        )
    return _merge(report, _load("selftest.json", {}))


def _run(argv: list[str]) -> int:
    values, flags, ref, command = _parse_run(argv)
    image = _image(ref)
    if image is None:
        sys.stderr.write(f"Unable to find image '{ref}' locally\n")
        return 125
    assert values.get("--pull") == ["never"], "the sandbox never pulls"
    labels = dict(label.split("=", 1) for label in values.get("--label", []))
    name = values.get("--name", [uuid.uuid4().hex[:12]])[0]
    container = {
        "Id": uuid.uuid4().hex * 2,
        "Name": name,
        "Rm": "--rm" in flags,
        "Pid": None if "-d" in flags else os.getpid(),
        "Config": {"Labels": labels, "Image": image["Id"]},
        "State": {"Running": True},
    }
    _save("containers.json", [*_containers(), container])
    if "-d" in flags:
        print(container["Id"])
        return 0
    if command[:3] == ["python", "-I", "-B"] and command[3:4] != ["-c"]:  # the deprecated bind transport
        mounts = dict(spec.rsplit(":", 1)[0].split(":", 1)[::-1] for spec in values.get("-v", []))
        runner = command[3].replace("/opt/orkexa/orkexa", mounts["/opt/orkexa/orkexa"], 1)
        os.chdir(mounts["/work"])
        os.environ.clear()
        os.environ.update({"PATH": "/usr/bin:/bin"})
        os.execv(sys.executable, [sys.executable, "-I", "-B", runner])
    assert command[:4] == ["python", "-I", "-B", "-c"], command
    script = command[4]
    if "ORKEXA_SELF_TEST" in script:
        sys.stdin.read()
        print(json.dumps(_self_test_report(values, flags, image)))
        return int(_load("selftest_exit.json", 0))
    work = tempfile.mkdtemp(prefix="fake-docker-work-")
    os.chdir(work)
    os.environ.clear()
    os.environ.update({"TMPDIR": tempfile.mkdtemp(prefix="fake-docker-tmp-"), "PATH": "/usr/bin:/bin"})
    os.execv(sys.executable, [sys.executable, "-I", "-B", "-c", script])
    return 1  # pragma: no cover


def _remove(ref: str, *, kill_only: bool = False) -> int:
    container = _find(ref)
    if container is None:
        sys.stderr.write(f"Error response from daemon: No such container: {ref}\n")
        return 1
    if _alive(container.get("Pid")):
        with contextlib.suppress(OSError):
            os.kill(container["Pid"], signal.SIGKILL)
    if not kill_only or container.get("Rm"):
        _save("containers.json", [c for c in _containers() if c["Id"] != container["Id"]])
    print(ref)
    return 0


def main() -> int:
    argv = sys.argv[1:]
    with open(_state() / "log.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps(argv) + "\n")
    failures = _load("fail.json", {})
    if argv and argv[0] in failures:
        sys.stderr.write(f"fake failure of {argv[0]}\n")
        return int(failures[argv[0]])
    host = _load("host.json", {})
    if argv[:3] == ["info", "--format", "{{json .}}"]:
        print(json.dumps(host.get("info")))
        return 0
    if argv[:3] == ["version", "--format", "{{json .}}"]:
        if host.get("version") is None:
            return 125
        print(json.dumps(host["version"]))
        return 0
    if argv[:2] == ["image", "inspect"]:
        found = [_image(ref) for ref in argv[2:]]
        print(json.dumps([image for image in found if image is not None]))
        return 0 if all(found) else 1
    if argv[:1] == ["pull"]:
        pullable = _load("pullable.json", {})
        if argv[1] not in pullable:
            sys.stderr.write("manifest unknown\n")
            return 1
        images = _load("images.json", {})
        images[argv[1]] = pullable[argv[1]]
        _save("images.json", images)
        return 0
    if argv[:1] == ["run"]:
        return _run(argv[1:])
    if argv[:1] == ["kill"]:
        return _remove(argv[1], kill_only=True)
    if argv[:2] == ["rm", "-f"]:
        return max((_remove(ref) for ref in argv[2:]), default=0)
    if argv[:2] == ["ps", "-a"]:
        wanted = [f.split("=", 1)[1] for f in argv[argv.index("--filter") + 1 : argv.index("--filter") + 2]]
        key, _, value = wanted[0].partition("=")
        for container in _containers():
            if container["Config"]["Labels"].get(key) == value:
                print(container["Id"])
        return 0
    if argv[:2] == ["container", "inspect"]:
        found = [_find(ref) for ref in argv[2:]]
        print(json.dumps([{**c, "Name": "/" + c["Name"]} for c in found if c is not None]))
        return 0 if all(found) else 1
    sys.stderr.write(f"fake docker: unsupported {argv}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
