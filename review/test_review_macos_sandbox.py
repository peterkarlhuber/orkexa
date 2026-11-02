"""Independent security review of the experimental MacOSSandbox (ORK-014).

Written independently of tests/test_macos_sandbox.py. Every probe runs real tool code in a confined call
and reports ALLOWED or refused; a probe that is expected to be refused fails the test when it is ALLOWED.
Measurements (overshoot of watchdogs, leaks that are documented) are printed and written to
``$REVIEW_OUT/<test>.json`` so the report can quote them. Nothing here changes the sandbox or its policy.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import pytest

from orkexa import ToolExecutionError
from orkexa.dynamic import MacOSSandbox, SandboxCancelled, cancel_scope
from orkexa.dynamic import macos as macos_module
from orkexa.dynamic.macos import _usage, footprint

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS")
OUT = Path(os.environ.get("REVIEW_OUT", "review-out"))
TOKEN = f"REVIEWSECRET{os.getpid()}"

HELPERS = r'''
import ctypes, ctypes.util, errno, json, os, socket, sys
c = json.loads(context)
lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
def rc(value):
    if value != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"rc={value} errno={e} {os.strerror(e) if e else ''}")
    return value
def attempt(action):
    try:
        return "ALLOWED: " + repr(action())[:300]
    except BaseException as exc:
        return "refused: " + type(exc).__name__ + ": " + str(exc)[:120]
def read(path):
    with open(path, "rb") as f:
        return f.read(200)
'''


def tool(body: str) -> str:
    lines = "\n".join("    " + line if line else "" for line in (HELPERS + body).strip("\n").splitlines())
    return (
        "def probe(context: str) -> str:\n"
        '    """Probe.\n\n    Args:\n        context: Input.\n\n    Returns:\n        Text.\n    """\n'
        f"{lines}\n"
    )


def save(name: str, data: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=str))
    print(f"\n=== {name} ===\n{json.dumps(data, indent=1, default=str)}")


def run(sandbox: MacOSSandbox, body: str, context: dict[str, Any]) -> dict[str, str]:
    return json.loads(sandbox.call(tool(body), "probe", {"context": json.dumps(context)}))


def escapes(results: dict[str, str], expected_allowed: tuple[str, ...] = ()) -> dict[str, str]:
    return {k: v for k, v in results.items() if v.startswith("ALLOWED") and k not in expected_allowed}


@pytest.fixture(scope="module")
def sandbox() -> MacOSSandbox:
    probe = MacOSSandbox.probe(refresh=True)
    save("probe", {"probe": probe.as_dict(), "uname": list(os.uname()), "python": sys.version})
    assert probe.available, probe
    return MacOSSandbox(timeout=20.0, memory_mb=512, tmp_mb=16)


@pytest.fixture(scope="module")
def secrets() -> Any:
    """Canary files with secret content in the places Nodestro data and credentials live."""
    home = Path.home()
    places = {
        "home": home / f".review-secret-{os.getpid()}",
        "app_support": home / "Library" / "Application Support" / "Nodestro" / "review-canary.txt",
        "ssh": home / ".ssh" / f"review_id_{os.getpid()}",
        "users_shared": Path("/Users/Shared") / f"review-canary-{os.getpid()}",
        "private_tmp": Path("/private/tmp") / f"review-canary-{os.getpid()}",
        "temp_root": Path(tempfile.gettempdir()) / f"review-canary-{os.getpid()}",
        "workspace_root": Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd())) / f".env-review-{os.getpid()}",
    }
    made = {}
    for name, path in places.items():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(TOKEN)
            made[name] = str(path)
        except OSError as exc:
            print(f"cannot create canary {name} at {path}: {exc}")
    with contextlib.suppress(OSError):
        os.setxattr(made["home"], "com.example.secret", TOKEN.encode())
    yield made
    for path in made.values():
        with contextlib.suppress(OSError):
            os.unlink(path)


def test_readable_paths_of_the_profile() -> None:
    """Which directories the profile makes readable on this runner (input for the report)."""
    code = (
        "import json, sys, tempfile, pathlib;"
        "sys.path.insert(0, sys.argv[1]);"
        "from orkexa.dynamic import _macos_confine as m;"
        "f=[str(pathlib.Path.home()), tempfile.gettempdir()];"
        "print(json.dumps({'interpreter_paths': m.interpreter_paths([__import__('os').path.realpath(p) for p in f]),"
        "'system': [p for p in m.SYSTEM_READ], 'sys_path': sys.path}))"
    )
    pkg_root = str(Path(macos_module.__file__).resolve().parents[2])
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code, pkg_root], capture_output=True, text=True)
    data = json.loads(out.stdout) if out.returncode == 0 else {"error": out.stderr}
    data["runner_package_dir"] = str(Path(macos_module._RUNNER).parents[1])
    save("readable_paths", data)


def test_files_secrets_and_bypasses(sandbox: MacOSSandbox, secrets: dict[str, str]) -> None:
    tmp_body = 'tmp = os.environ["TMPDIR"]\n'
    body = tmp_body + r'''
r = {}
for name, path in c["secrets"].items():
    r["read " + name] = attempt(lambda p=path: read(p))
home = c["home"]
r["listdir home"] = attempt(lambda: os.listdir(home))
r["listdir /Users"] = attempt(lambda: os.listdir("/Users"))
r["listdir temp root"] = attempt(lambda: os.listdir(c["temp_root"]))
r["listdir /Applications"] = attempt(lambda: os.listdir("/Applications"))
r["read /etc/passwd"] = attempt(lambda: read("/etc/passwd"))
r["read keychain"] = attempt(lambda: read(home + "/Library/Keychains/login.keychain-db"))
r["read TCC db"] = attempt(lambda: read(home + "/Library/Application Support/com.apple.TCC/TCC.db"))
s = c["secrets"]["home"]
def hardlink():
    os.link(s, tmp + "/hl"); return read(tmp + "/hl")
r["hardlink secret into tmp"] = attempt(hardlink)
def symlink():
    os.symlink(s, tmp + "/sl"); return read(tmp + "/sl")
r["symlink secret into tmp"] = attempt(symlink)
def clone():
    rc(lib.clonefile(s.encode(), (tmp + "/cl").encode(), 0)); return read(tmp + "/cl")
r["clonefile secret into tmp"] = attempt(clone)
def rename_in():
    os.rename(s, tmp + "/mv"); return read(tmp + "/mv")
r["rename secret into tmp"] = attempt(rename_in)
def swap():
    open(tmp + "/sw", "w").write("x")
    rc(lib.renamex_np((tmp + "/sw").encode(), s.encode(), 2)); return read(tmp + "/sw")
r["renamex_np swap secret into tmp"] = attempt(swap)
def exchange():
    open(tmp + "/ex", "w").write("x")
    rc(lib.exchangedata((tmp + "/ex").encode(), s.encode(), 0)); return read(tmp + "/ex")
r["exchangedata secret"] = attempt(exchange)
r["getxattr secret"] = attempt(lambda: os.getxattr(s, "com.example.secret"))
r["write home"] = attempt(lambda: open(home + "/review-escape.txt", "w").write("x"))
r["write /private/tmp"] = attempt(lambda: open("/private/tmp/review-escape.txt", "w").write("x"))
r["write site-packages (persistence)"] = attempt(lambda: open(c["site"] + "/review_escape.pth", "w").write("x"))
r["write orkexa package (persistence)"] = attempt(lambda: open(c["pkg"] + "/review_escape.py", "w").write("x"))
r["write call dir outside tmp"] = attempt(lambda: open(os.getcwd() + "/tool.py", "a").write("x"))
r["unlink secret"] = attempt(lambda: os.unlink(c["secrets"]["users_shared"]))
r["chmod secret"] = attempt(lambda: os.chmod(s, 0o777))
r["utime secret"] = attempt(lambda: os.utime(s, (0, 0)))
r["setxattr secret"] = attempt(lambda: os.setxattr(s, "com.example.x", b"x"))
r["truncate secret"] = attempt(lambda: os.truncate(s, 0))
r["mkdir home"] = attempt(lambda: os.mkdir(home + "/review-escape-dir"))
r["open /dev/tty"] = attempt(lambda: open("/dev/tty", "w"))
r["load dylib from tmp"] = attempt(lambda: (open(tmp + "/x.dylib", "wb").write(open("/usr/lib/libSystem.B.dylib", "rb").read(1)), ctypes.CDLL(tmp + "/x.dylib")))
# documented metadata leak (expected to be allowed):
r["stat secret (metadata)"] = attempt(lambda: os.stat(s).st_size)
r["exists ~/.ssh (metadata)"] = attempt(lambda: os.path.isdir(home + "/.ssh"))
return json.dumps(r)
'''
    import site

    context = {
        "secrets": secrets,
        "home": str(Path.home()),
        "temp_root": tempfile.gettempdir(),
        "site": site.getsitepackages()[0],
        "pkg": str(Path(macos_module.__file__).resolve().parent),
    }
    results = run(sandbox, body, context)
    leaked = [k for k, v in results.items() if TOKEN in v]
    save("files", {"results": results, "token_leaked_in": leaked})
    for path in (Path.home() / "review-escape.txt", Path("/private/tmp/review-escape.txt")):
        assert not path.exists(), path
    assert all(Path(p).exists() for p in secrets.values())
    assert not leaked, leaked
    expected = ("stat secret (metadata)", "exists ~/.ssh (metadata)")
    assert not escapes(results, expected), escapes(results, expected)


@pytest.fixture()
def helper_process() -> Any:
    """A host process (same user) with a secret in its argv and environment."""
    env = {"PATH": "/usr/bin:/bin", "REVIEW_ENV_SECRET": TOKEN}
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)", f"--token={TOKEN}"], env=env)
    time.sleep(0.5)
    yield proc
    proc.kill()
    proc.wait()


def test_processes_mach_ipc_and_kernel_surface(sandbox: MacOSSandbox, helper_process: Any) -> None:
    body = r'''
r = {}
pid = c["pid"]; parent = os.getppid()
def procargs(p):
    mib = (ctypes.c_int * 3)(1, 49, p)  # CTL_KERN, KERN_PROCARGS2
    size = ctypes.c_size_t(1 << 18); buf = ctypes.create_string_buffer(size.value)
    rc(lib.sysctl(mib, 3, buf, ctypes.byref(size), None, 0))
    return buf.raw[: size.value].replace(b"\x00", b" ")[-400:]
r["KERN_PROCARGS2 other process (argv+env)"] = attempt(lambda: procargs(pid))
r["KERN_PROCARGS2 parent (host)"] = attempt(lambda: procargs(parent))
def proclist():
    mib = (ctypes.c_int * 3)(1, 14, 0)  # KERN_PROC_ALL
    size = ctypes.c_size_t(0)
    rc(lib.sysctl(mib, 3, None, ctypes.byref(size), None, 0))
    return size.value
r["KERN_PROC_ALL size (process list)"] = attempt(proclist)
def listpids():
    buf = (ctypes.c_int * 8192)()
    n = lib.proc_listallpids(buf, ctypes.sizeof(buf))
    if n <= 0: raise OSError(ctypes.get_errno(), "proc_listallpids")
    return n
r["proc_listallpids"] = attempt(listpids)
def pidpath(p):
    buf = ctypes.create_string_buffer(4096)
    n = lib.proc_pidpath(p, buf, 4096)
    if n <= 0: raise OSError(ctypes.get_errno(), "proc_pidpath")
    return buf.value
r["proc_pidpath other"] = attempt(lambda: pidpath(pid))
def pidinfo(p):
    buf = ctypes.create_string_buffer(1024)
    n = lib.proc_pidinfo(p, 3, 0, buf, 1024)  # PROC_PIDTBSDINFO
    if n <= 0: raise OSError(ctypes.get_errno(), "proc_pidinfo")
    return n
r["proc_pidinfo other"] = attempt(lambda: pidinfo(pid))
r["kill(other, 0)"] = attempt(lambda: os.kill(pid, 0))
r["kill(other, SIGCONT)"] = attempt(lambda: os.kill(pid, 18))
r["kill(parent, SIGCONT)"] = attempt(lambda: os.kill(parent, 18))
r["kill(-1, SIGCONT) all user processes"] = attempt(lambda: os.kill(-1, 18))
def tfp(p):
    me = ctypes.c_uint.in_dll(lib, "mach_task_self_").value
    port = ctypes.c_uint(0)
    kr = lib.task_for_pid(me, p, ctypes.byref(port))
    if kr != 0: raise OSError(kr, f"kern_return {kr}")
    return port.value
r["task_for_pid other"] = attempt(lambda: tfp(pid))
r["task_for_pid parent"] = attempt(lambda: tfp(parent))
def task_name(p):
    me = ctypes.c_uint.in_dll(lib, "mach_task_self_").value
    port = ctypes.c_uint(0)
    kr = lib.task_name_for_pid(me, p, ctypes.byref(port))
    if kr != 0: raise OSError(kr, f"kern_return {kr}")
    return port.value
r["task_name_for_pid other"] = attempt(lambda: task_name(pid))
def ptrace_attach(p):
    rc(lib.ptrace(14, p, None, 0))  # PT_ATTACHEXC
    lib.ptrace(11, p, None, 0)       # PT_DETACH
    return "attached"
r["ptrace attach other"] = attempt(lambda: ptrace_attach(pid))
boot = ctypes.c_uint.in_dll(lib, "bootstrap_port").value
for service in ("com.apple.pasteboard.1", "com.apple.SecurityServer", "com.apple.securityd.xpc",
                "com.apple.coreservices.launchservicesd", "com.apple.lsd.mapdb", "com.apple.windowserver.active",
                "com.apple.distributed_notifications@Uv3", "com.apple.FileCoordination", "com.apple.tccd",
                "com.apple.system.opendirectoryd.libinfo", "com.apple.dnssd.service", "com.apple.xpc.launchd.domain.system",
                "com.apple.logd", "com.apple.system.notification_center", "com.apple.cfprefsd.daemon",
                "com.apple.cfprefsd.agent", "com.apple.trustd", "com.apple.nsurlsessiond"):
    def look(name=service):
        port = ctypes.c_uint(0)
        kr = lib.bootstrap_look_up(boot, name.encode(), ctypes.byref(port))
        if kr != 0: raise OSError(kr, f"kern_return {kr}")
        return port.value
    r["mach-lookup " + service] = attempt(look)
def shm():
    lib.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    fd = lib.shm_open(b"/review-shm", 0x200 | 2, 0o600)
    if fd < 0: raise OSError(ctypes.get_errno(), "shm_open")
    lib.shm_unlink(b"/review-shm"); return fd
r["posix shm_open"] = attempt(shm)
def sem():
    lib.sem_open.restype = ctypes.c_void_p
    lib.sem_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    h = lib.sem_open(b"/review-sem", 0x200, 0o600, 1)
    if h in (None, ctypes.c_void_p(-1).value): raise OSError(ctypes.get_errno(), "sem_open")
    lib.sem_unlink(b"/review-sem"); return "ok"
r["posix sem_open"] = attempt(sem)
def sysv():
    i = lib.shmget(0, 4096, 0o1000 | 0o600)
    if i < 0: raise OSError(ctypes.get_errno(), "shmget")
    return i
r["sysv shmget"] = attempt(sysv)
def iokit(name):
    io = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    io.IOServiceMatching.restype = ctypes.c_void_p
    io.IOServiceGetMatchingService.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    svc = io.IOServiceGetMatchingService(0, io.IOServiceMatching(name.encode()))
    if not svc: raise OSError(0, "no service")
    conn = ctypes.c_uint(0)
    me = ctypes.c_uint.in_dll(lib, "mach_task_self_").value
    kr = io.IOServiceOpen(svc, me, 0, ctypes.byref(conn))
    if kr != 0: raise OSError(kr, f"IOServiceOpen {kr:#x}")
    return conn.value
r["IOServiceOpen IOSurfaceRoot"] = attempt(lambda: iokit("IOSurfaceRoot"))
r["IOServiceOpen AppleKeyStore"] = attempt(lambda: iokit("AppleKeyStore"))
r["fork"] = attempt(lambda: os.fork())
r["posix_spawn"] = attempt(lambda: os.posix_spawn("/bin/echo", ["echo"], {}))
import resource
r["raise RLIMIT_NPROC"] = attempt(lambda: resource.setrlimit(resource.RLIMIT_NPROC, (10, 10)))
r["raise RLIMIT_CPU"] = attempt(lambda: resource.setrlimit(resource.RLIMIT_CPU, (10**6, 10**6)))
r["open fds > 2"] = "ALLOWED: " + repr([fd for fd in range(3, 256) if not attempt(lambda f=fd: os.fstat(f)).startswith("refused")]) if any(not attempt(lambda f=fd: os.fstat(f)).startswith("refused") for fd in range(3, 256)) else "refused: none"
r["env"] = "ALLOWED: " + repr(sorted(os.environ)) if c["token"] in repr(dict(os.environ)) else "refused: clean (" + ",".join(sorted(os.environ)) + ")"
# documented (allowed): sysctl-read
def sysctl_name(name):
    size = ctypes.c_size_t(256); buf = ctypes.create_string_buffer(256)
    rc(lib.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0)); return buf.value
r["sysctl kern.hostname (info)"] = attempt(lambda: sysctl_name("kern.hostname"))
r["sysctl kern.osversion (info)"] = attempt(lambda: sysctl_name("kern.osversion"))
return json.dumps(r)
'''
    results = run(sandbox, body, {"pid": helper_process.pid, "token": TOKEN})
    leaked = [k for k, v in results.items() if TOKEN in v]
    save("processes", {"results": results, "token_leaked_in": leaked})
    assert helper_process.poll() is None, "the helper process was affected"
    assert not leaked, leaked
    expected = ("sysctl kern.hostname (info)", "sysctl kern.osversion (info)", "KERN_PROC_ALL size (process list)",
                "proc_listallpids")
    assert not escapes(results, expected), escapes(results, expected)


def test_exec_replacing_the_process_is_refused(sandbox: MacOSSandbox) -> None:
    body = r'''
try:
    os.execv("/bin/sh", ["/bin/sh", "-c", "printf '{\"ok\": true, \"result\": \"EXEC\"}'"])
except OSError as exc:
    return "refused: " + str(exc)
'''
    try:
        result = sandbox.call(tool(body), "probe", {"context": "{}"})
    except ToolExecutionError as exc:
        result = f"error: {exc}"
    save("exec", {"result": result})
    assert result != "EXEC"
    assert result.startswith("refused")


def test_network(sandbox: MacOSSandbox) -> None:
    v4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    v4.bind(("127.0.0.1", 0))
    v4.listen(4)
    v6 = None
    with contextlib.suppress(OSError):
        v6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        v6.bind(("::1", 0))
        v6.listen(4)
    unix_dir = tempfile.mkdtemp(prefix="review-unix-", dir="/private/tmp")
    unix_path = os.path.join(unix_dir, "s")
    ux = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ux.bind(unix_path)
    ux.listen(4)
    body = r'''
r = {}
r["tcp 127.0.0.1"] = attempt(lambda: socket.create_connection(("127.0.0.1", c["v4"]), timeout=2))
if c["v6"]:
    r["tcp ::1"] = attempt(lambda: socket.create_connection(("::1", c["v6"]), timeout=2))
r["tcp 1.1.1.1:443"] = attempt(lambda: socket.create_connection(("1.1.1.1", 443), timeout=3))
r["tcp 140.82.112.3:443"] = attempt(lambda: socket.create_connection(("140.82.112.3", 443), timeout=3))
def udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); return s.sendto(b"x", ("1.1.1.1", 53))
r["udp sendto 1.1.1.1:53"] = attempt(udp)
def icmp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_ICMP)
    return s.sendto(b"\x08\x00\xf7\xff\x00\x00\x00\x00", ("1.1.1.1", 0))
r["icmp dgram sendto"] = attempt(icmp)
r["dns getaddrinfo"] = attempt(lambda: socket.getaddrinfo("example.com", 443))
def unix(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(path); return "connected"
r["unix host socket"] = attempt(lambda: unix(c["unix"]))
r["unix mDNSResponder"] = attempt(lambda: unix("/var/run/mDNSResponder"))
r["unix syslog"] = attempt(lambda: unix("/var/run/syslog"))
def listen():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1); return s.getsockname()
r["tcp listen"] = attempt(listen)
def bind_unix():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.bind(os.environ["TMPDIR"] + "/u"); s.listen(1); return "bound"
r["unix listen in tmp"] = attempt(bind_unix)
def route():
    s = socket.socket(socket.AF_ROUTE, socket.SOCK_RAW, 0); return "opened"
r["PF_ROUTE socket"] = attempt(route)
def sysctl_net():
    mib = (ctypes.c_int * 6)(4, 17, 0, 0, 3, 0)  # CTL_NET, PF_ROUTE, NET_RT_IFLIST
    size = ctypes.c_size_t(0)
    rc(lib.sysctl(mib, 6, None, ctypes.byref(size), None, 0)); return size.value
r["sysctl NET_RT_IFLIST (info)"] = attempt(sysctl_net)
return json.dumps(r)
'''
    context = {"v4": v4.getsockname()[1], "v6": v6.getsockname()[1] if v6 else 0, "unix": unix_path}
    accepted: list[str] = []
    v4.settimeout(0.1)

    try:
        results = run(sandbox, body, context)
        with contextlib.suppress(OSError):
            v4.accept()
            accepted.append("v4")
    finally:
        v4.close()
        if v6:
            v6.close()
        ux.close()
        shutil.rmtree(unix_dir, ignore_errors=True)
    save("network", {"results": results, "host_accepted": accepted})
    assert not accepted
    expected = ("sysctl NET_RT_IFLIST (info)",)
    assert not escapes(results, expected), escapes(results, expected)


class _Recorder:
    """Records the pids of processes that run_bounded starts."""

    pids: list[int] = []


@pytest.fixture()
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    real = subprocess.Popen
    pids: list[int] = []

    class Popen(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            pids.append(self.pid)

    monkeypatch.setattr(subprocess, "Popen", Popen)
    return pids


def _poll_peak(pids: list[int], stop: threading.Event, peak: dict[str, int]) -> None:
    while not stop.is_set():
        for pid in list(pids):
            peak["footprint"] = max(peak.get("footprint", 0), footprint(pid))
        for d in glob.glob(os.path.join(tempfile.gettempdir(), "orkexa-sandbox-*", "tmp")):
            size, count = _usage(Path(d))
            peak["tmp_bytes"] = max(peak.get("tmp_bytes", 0), size)
            peak["tmp_entries"] = max(peak.get("tmp_entries", 0), count)
        time.sleep(0.002)


def _measured(sandbox: MacOSSandbox, body: str, pids: list[int]) -> dict[str, Any]:
    stop = threading.Event()
    peak: dict[str, int] = {}
    poller = threading.Thread(target=_poll_peak, args=(pids, stop, peak), daemon=True)
    poller.start()
    started = time.monotonic()
    try:
        result = "returned: " + sandbox.call(tool(body), "probe", {"context": "{}"})
    except ToolExecutionError as exc:
        result = f"stopped: {exc}"
    except Exception as exc:  # anything else escapes the sandbox API contract
        result = f"UNEXPECTED {type(exc).__name__}: {exc}"
    stop.set()
    poller.join()
    return {"result": result[:400], "seconds": round(time.monotonic() - started, 2), **peak}


def test_memory_burst_overshoot(recorded: list[int]) -> None:
    sandbox = MacOSSandbox(timeout=20.0, memory_mb=256, tmp_mb=16)
    gradual = _measured(sandbox, "blocks = []\nwhile True:\n    blocks.append(b'\\x01' * (64 << 20))", recorded)
    burst = _measured(
        sandbox,
        "import mmap\nn = 3 << 30\nm = mmap.mmap(-1, n)\n"
        "addr = ctypes.addressof(ctypes.c_char.from_buffer(m))\nctypes.memset(addr, 1, n)\nreturn 'survived'",
        recorded,
    )
    data = {"limit_mb": 256, "gradual": gradual, "burst_3gb": burst}
    for v in (gradual, burst):
        v["peak_mb"] = round(v.get("footprint", 0) / 2**20)
    save("memory", data)
    assert gradual["result"].startswith("stopped") and burst["result"].startswith("stopped"), data


def test_disk_watchdog(recorded: list[int]) -> None:
    sandbox = MacOSSandbox(timeout=20.0, memory_mb=512, tmp_mb=16)
    fast = _measured(
        sandbox,
        "tmp = os.environ['TMPDIR']\nblock = b'x' * (1 << 20)\ni = 0\nwhile i < 2000:\n"
        "    open(f'{tmp}/f{i}', 'wb').write(block); i += 1\nreturn 'wrote 2000 MB'",
        recorded,
    )
    many = _measured(
        sandbox,
        "tmp = os.environ['TMPDIR']\nfor i in range(200000):\n    open(f'{tmp}/e{i}', 'wb').close()\nreturn 'made 200000'",
        recorded,
    )
    # A directory the host cannot list (write+search, no read): the watchdog's scandir fails and is skipped.
    hidden = _measured(
        sandbox,
        "tmp = os.environ['TMPDIR']\nd = tmp + '/hidden'\nos.mkdir(d)\nos.chmod(d, 0o300)\n"
        "block = b'x' * (1 << 20)\nfor i in range(200):\n    open(f'{d}/f{i}', 'wb').write(block)\n"
        "return 'wrote 200 MB into a hidden directory'",
        recorded,
    )
    data = {"limit_mb": 16, "limit_entries": macos_module.MAX_TMP_ENTRIES, "fast_writes": fast,
            "many_entries": many, "unlistable_dir": hidden}
    save("disk", data)
    assert fast["result"].startswith("stopped"), data
    assert many["result"].startswith("stopped"), data
    assert hidden["result"].startswith("stopped"), data


def test_cleanup_with_immutable_and_locked_files(recorded: list[int]) -> None:
    sandbox = MacOSSandbox(timeout=20.0, memory_mb=512, tmp_mb=16)
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "orkexa-sandbox-*")))
    body = (
        "tmp = os.environ['TMPDIR']\nopen(tmp + '/locked', 'w').write('x')\n"
        "lib.chflags.argtypes = [ctypes.c_char_p, ctypes.c_uint]\n"
        "r = attempt(lambda: rc(lib.chflags((tmp + '/locked').encode(), 0x2)))\n"  # UF_IMMUTABLE
        "os.mkdir(tmp + '/d'); open(tmp + '/d/f', 'w').write('x'); os.chmod(tmp + '/d', 0)\n"
        "return 'chflags ' + r"
    )
    result = _measured(sandbox, body, recorded)
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "orkexa-sandbox-*")))
    left = sorted(after - before)
    save("cleanup", {"call": result, "left_behind": left})
    for path in left:  # clean up for the next tests
        subprocess.run(["chflags", "-R", "nouchg", path], check=False)
        subprocess.run(["chmod", "-R", "u+rwx", path], check=False)
        shutil.rmtree(path, ignore_errors=True)
    assert not result["result"].startswith("UNEXPECTED"), result
    assert not left, left


def test_cpu_threads_and_wall_clock(recorded: list[int]) -> None:
    sandbox = MacOSSandbox(timeout=5.0, memory_mb=512, tmp_mb=16)
    spin = (
        "import hashlib, threading\nblob = b'x' * (64 << 20)\n"
        "def burn():\n    while True:\n        hashlib.sha256(blob).digest()\n"
        "for _ in range(16):\n    threading.Thread(target=burn, daemon=True).start()\n"
        "import time\ntime.sleep(60)\nreturn 'never'"
    )
    cpu = _measured(sandbox, spin, recorded)
    threads = _measured(
        sandbox,
        "import threading, time\nn = 0\ntry:\n    while n < 20000:\n"
        "        threading.Thread(target=time.sleep, args=(30,), daemon=True).start(); n += 1\n"
        "except BaseException as exc:\n    return f'threads {n}: {type(exc).__name__}'\nreturn f'threads {n}'",
        recorded,
    )
    alive = [pid for pid in recorded if _alive(pid)]
    save("cpu_threads", {"cpu_16_threads": cpu, "thread_bomb": threads, "processes_still_alive": alive})
    assert cpu["result"].startswith("stopped") and cpu["seconds"] < 15, cpu
    assert not alive, alive


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    with contextlib.suppress(ChildProcessError):
        if os.waitpid(pid, os.WNOHANG) != (0, 0):
            return False
    return True


def test_parallel_calls_isolation_cancel_and_cleanup(recorded: list[int]) -> None:
    sandbox = MacOSSandbox(timeout=15.0, memory_mb=256, tmp_mb=16)
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "orkexa-sandbox-*")))
    writer = tool(
        "tmp = os.environ['TMPDIR']\nopen(tmp + '/mine.txt', 'w').write(c['token'])\n"
        "import time\ntime.sleep(6)\nreturn 'writer done'"
    )
    results: dict[str, str] = {}

    def go(name: str, code: str, context: dict[str, Any], cancel_after: float | None = None) -> None:
        stop = threading.Event()
        if cancel_after is not None:
            threading.Timer(cancel_after, stop.set).start()
        try:
            with cancel_scope(stop.is_set):
                results[name] = sandbox.call(code, "probe", {"context": json.dumps(context)})
        except ToolExecutionError as exc:
            results[name] = f"stopped: {type(exc.__cause__).__name__}: {exc}"[:200]

    threads = [threading.Thread(target=go, args=("writer", writer, {"token": TOKEN}))]
    threads[0].start()
    time.sleep(2.0)
    files = glob.glob(os.path.join(tempfile.gettempdir(), "orkexa-sandbox-*", "tmp", "mine.txt"))
    writer_pids = list(recorded)
    reader = tool(
        "r = {}\nfor p in c['files']:\n    r['read ' + p[-40:]] = attempt(lambda p=p: read(p))\n"
        "    r['write ' + p[-40:]] = attempt(lambda p=p: open(p, 'a').write('x'))\n"
        "for pid in c['pids']:\n    r[f'kill {pid}'] = attempt(lambda pid=pid: os.kill(pid, 18))\n"
        "return json.dumps(r)"
    )
    sleeper = tool("import time\ntime.sleep(30)\nreturn 'never'")
    for i in range(6):
        threads.append(threading.Thread(target=go, args=(f"reader{i}", reader, {"files": files, "pids": writer_pids})))
    for i in range(6):
        threads.append(threading.Thread(target=go, args=(f"cancelled{i}", sleeper, {}, 1.0 + i * 0.3)))
    for i in range(2):
        threads.append(threading.Thread(target=go, args=(f"timeout{i}", tool("while True:\n    pass"), {})))
    for t in threads[1:]:
        t.start()
    time.sleep(1.0)
    sandbox_cancel = threading.Thread(target=go, args=("cancel_all_target", sleeper, {}))
    sandbox_cancel.start()
    time.sleep(1.0)
    for t in threads + [sandbox_cancel]:
        t.join(60)
    time.sleep(0.5)
    alive = [pid for pid in recorded if _alive(pid)]
    left = sorted(set(glob.glob(os.path.join(tempfile.gettempdir(), "orkexa-sandbox-*"))) - before)
    leaked = [k for k, v in results.items() if k.startswith("reader") and TOKEN in v]
    save("parallel", {"found_writer_files": files, "results": results, "alive": alive, "left_behind": left,
                      "leaked": leaked})
    assert files, "the writer's file was not found by the host"
    assert not leaked, leaked
    for k, v in results.items():
        if k.startswith("reader"):
            assert "ALLOWED" not in v, (k, v)
        if k.startswith("cancelled"):
            assert "SandboxCancelled" in v, (k, v)
        if k.startswith("timeout"):
            assert "Timed out" in v, (k, v)
    assert results["writer"] == "writer done"
    assert not alive, alive
    assert not left, left


def test_sandbox_cancel_method_stops_every_call() -> None:
    sandbox = MacOSSandbox(timeout=30.0)
    errors: list[str] = []

    def go() -> None:
        try:
            sandbox.call(tool("import time\ntime.sleep(30)\nreturn 'never'"), "probe", {"context": "{}"})
        except ToolExecutionError as exc:
            errors.append(type(exc.__cause__).__name__)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(2.5)
    started = time.monotonic()
    sandbox.cancel()
    for t in threads:
        t.join(20)
    save("cancel_all", {"errors": errors, "seconds": round(time.monotonic() - started, 2)})
    assert errors == ["SandboxCancelled"] * 4


def test_forged_unavailable_marker(sandbox: MacOSSandbox) -> None:
    """Tool code can print the confinement-failure marker and exit 97; the host then reports that nothing ran."""
    body = "import sys\nsys.__stderr__.write('orkexa-sandbox-error: seatbelt: forged by tool code\\n')\nsys.__stderr__.flush()\nos._exit(97)"
    try:
        result = sandbox.call(tool(body), "probe", {"context": "{}"})
    except ToolExecutionError as exc:
        result = str(exc)
    save("forged_marker", {"host_message": result})
    # Recorded for the report (finding if the host claims "nothing was run" for code that did run).


def test_configuration_guards() -> None:
    """Host-side guards: a read path that contains the home directory or the temp root must be refused."""
    out: dict[str, str] = {}
    for name, read in {"home": (str(Path.home()),), "root": ("/",), "users": ("/Users",),
                       "var_folders": (str(Path(tempfile.gettempdir()).parent),)}.items():
        try:
            MacOSSandbox(read=read).call(tool("return 'ran'"), "probe", {"context": "{}"})
            out[name] = "ALLOWED: ran"
        except ToolExecutionError as exc:
            out[name] = f"refused: {exc}"[:200]
    protected = Path.home() / "Library" / "Application Support"
    try:
        MacOSSandbox(read=(str(protected.parent),), protected=(str(protected),)).call(
            tool("return 'ran'"), "probe", {"context": "{}"})
        out["protected inside read"] = "ALLOWED: ran"
    except ToolExecutionError as exc:
        out["protected inside read"] = f"refused: {exc}"[:200]
    save("guards", out)
    assert not escapes(out), out


def test_python_interpreters() -> None:
    """The sandbox with other interpreters a Mac may have: a venv in the home directory and /usr/bin/python3."""
    out: dict[str, Any] = {}
    venv = Path.home() / f"review-venv-{os.getpid()}"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
    candidates = {"venv_in_home": str(venv / "bin" / "python")}
    canary = Path.home() / f".review-interp-{os.getpid()}"
    canary.write_text(TOKEN)
    try:
        for name, python in candidates.items():
            sb = MacOSSandbox(python=python, timeout=30.0)
            try:
                out[name] = run(sb, 'r = {"home canary": attempt(lambda: read(c["canary"])), "exe": "info: " + sys.executable,'
                                    ' "version": "info: " + sys.version.split()[0]}\nreturn json.dumps(r)', {"canary": str(canary)})
            except ToolExecutionError as exc:
                out[name] = {"error": str(exc)[:300]}
    finally:
        canary.unlink()
        shutil.rmtree(venv, ignore_errors=True)
    save("interpreters", out)
    for name, value in out.items():
        if isinstance(value, dict) and "home canary" in value:
            assert value["home canary"].startswith("refused"), (name, value)
