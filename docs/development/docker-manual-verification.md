# Docker manual verification

The `docker` jobs of the `Platforms` workflow check rootful and rootless Docker, rootful
Podman, Linux arm64 and (non-blocking) Colima on macOS x86-64 on every push, and that
rootless Podman 4.9.3 is refused (it does not apply the `--ulimit` limits). Docker
Desktop has never run in CI: hosted runners cannot run it on Apple Silicon (no nested
virtualization), Linux containers on Windows (x64 or ARM64), OrbStack or `podman
machine`, so these hosts are **unverified in CI** (see the
[feature matrix](../feature-matrix.md#docker-sandbox-hosts)). Use this checklist on such
a computer before relying on it, after a Docker or OS update, and before a release.

## What you need

* Python 3.11+ with Orkexa installed (`pip install -r requirements/tabrion.txt -e ".[dev]"` in a checkout).
* The engine under test running, with the `docker` CLI pointing at it (`docker context
  ls`, or `DOCKER_HOST`).
* The sandbox image: `docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .`

## Checklist

1. **Record the host.**

    ```bash
    docker version
    docker info
    python -c "import platform; print(platform.platform(), platform.machine())"
    ```

2. **Preflight.** It must print `docker: available`; otherwise note the `reason`
   (`os-type`, `limits`, `seccomp`, `rootless-cgroup`) and stop: the sandbox refuses this
   host by design.

    ```bash
    python -m orkexa.dynamic.docker --preflight
    ```

3. **Self-test.** It runs one container of exactly the sandbox's shape. It must print
   `docker: available` and `self_test: passed`; keep the JSON output for the record.

    ```bash
    python -m orkexa.dynamic.docker --self-test
    python -m orkexa.dynamic.docker --self-test --json > docker-self-test.json
    ```

    A refusal with reason `self-test` lists every check that failed (for example
    `Seccomp is 0, not 2`, `memory.max is max`, `the container is not on cgroup v2`,
    `unexpected writable mount at /data` for a volume from the engine's defaults,
    `the seccomp profile allows move_pages` for a permissive custom profile, or
    `the host reports AppArmor, but the container is not confined`). The fact
    `apparmor_label` shows the container's AppArmor profile (`not available` without
    AppArmor). Do not work around it: report the host and the output.

4. **Real-container tests.** They must all pass, none may be skipped:

    ```bash
    ORKEXA_REQUIRE_DOCKER=1 pytest -v -rs -m docker tests/test_docker_sandbox.py tests/anthropic/test_dynamic_sandbox.py
    ```

    They cover the self-test, a fork bomb against the process limit, an allocation of twice
    the memory limit (OOM-killed, exit code 137), network connections, writes to `/`, the
    file-size limit, cancellation (no container left) and the reaper.

5. **Nothing is left behind.** This must print nothing:

    ```bash
    docker ps -a --filter label=org.orkexa.sandbox
    ```

6. **A refused host stays refused.** Optional, where the engine allows it: turn off a
   prerequisite (for example the seccomp default with `"seccomp-profile": "unconfined"`
   in `daemon.json`, or cgroup delegation for rootless engines), restart the daemon and
   repeat step 2 or 3. Both must refuse, and
   `python -c "from orkexa.dynamic import DockerSandbox; print(DockerSandbox.available())"`
   must print `False`. Restore the setting afterwards.

## Notes per host

| Host | What to watch |
| --- | --- |
| Docker Desktop, macOS arm64 | Some versions report `seccomp,profile=unconfined`; the preflight then refuses (`seccomp`). Containers run in Docker Desktop's Linux VM, so the kernel is the VM's. |
| Docker Desktop, Windows (WSL2) | Linux containers only: "Switch to Windows containers" makes the daemon report `OSType: windows`, which is refused (`os-type`). Run the checklist from a WSL2 shell or PowerShell with Python for Windows; both use the same daemon. |
| Colima (macOS arm64 or x86-64) | Start with `colima start --vm-type vz` (or `qemu`); the CLI's socket is `~/.colima/default/docker.sock`. |
| OrbStack | Its daemon reports `OperatingSystem: OrbStack`; the name is only recorded, the checks decide. |
| Podman (Linux, macOS `podman machine`) | Use the Docker API (`podman system service`, or the machine's socket) through `DOCKER_HOST`, or pass `--docker podman`. Rootless Podman needs cgroup v2 and the `cpu`, `memory` and `pids` controllers delegated to the user. A `mounts.conf` entry (RHEL subscription secrets on `/run/secrets`), a containers.conf `volumes` or `tz` default is refused (`self-test`, "unexpected ... mount"): remove it for the sandbox's engine; the check is not relaxed. |

## Recording the result

Add the host, the engine version, the date and the outcome of steps 2 to 6 to the pull
request or issue that asked for the verification, with `docker-self-test.json` attached.
