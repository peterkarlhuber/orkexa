"""TEMPORARY (W3-S, removed before handover): repeat the conformance swap race and count its outcomes.

Runs the ``folders.swap_race`` case's two races (a file, a folder) ``N`` times with the case's own
parameters against the reference folder access of the conformance tests, and prints how often each finding
occurred, how many reads returned the regular file, and why the others were refused.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "conformance"))

from folder_access import SafeRootFolders

from orkexa.testing import pathguard


def main(rounds: int) -> int:
    findings: Counter[str] = Counter()
    reads: dict[bool, Counter[str]] = {False: Counter(), True: Counter()}
    zero_inside = Counter()
    for number in range(rounds):
        with tempfile.TemporaryDirectory() as work:
            access = SafeRootFolders(Path(work) / "data")
            for folder in (False, True):
                layout = pathguard.build_layout(Path(work) / f"race-{int(folder)}-{number}")
                grant = access.grant(str(layout.root))
                seen = reads[folder]
                before = seen["ok"]

                def read(
                    path: str, seen: Counter[str] = seen, access: SafeRootFolders = access, grant: str = grant
                ) -> bytes:
                    try:
                        data = access.read(grant, path).encode("utf-8")
                    except Exception as exc:
                        seen[getattr(exc, "code", None) or type(exc).__name__] += 1
                        raise
                    seen["ok"] += 1
                    return data

                found = pathguard.swap_race(read, layout, iterations=300, folder=folder, min_swaps=3)
                for item in found:
                    findings[f"folder={folder}: {item.problem}"] += 1
                if seen["ok"] == before:
                    zero_inside[folder] += 1
    print(f"rounds: {rounds}")
    print(f"findings: {dict(findings) or 'none'}")
    for folder, seen in reads.items():
        print(f"folder={folder}: reads {dict(seen)}; rounds without a regular read: {zero_inside[folder]}")
    return 1 if findings else 0


def vulnerable_rates(rounds: int) -> None:
    """How often a check-then-open reader is caught: master's swap_race and this branch's (#116 review)."""
    import importlib.util
    import os
    import stat

    from orkexa.testing import pathguard as branch

    source = Path(__file__).with_name("_pathguard_master.py")
    spec = importlib.util.spec_from_file_location("pathguard_master", source)
    assert spec is not None and spec.loader is not None
    master = importlib.util.module_from_spec(spec)
    sys.modules["pathguard_master"] = master
    spec.loader.exec_module(master)
    for label, module in (("master", master), ("branch", branch), ("master", master), ("branch", branch)):
        caught = clean = other = 0
        for number in range(rounds):
            with tempfile.TemporaryDirectory() as work:
                layout = module.build_layout(Path(work) / f"v{number}")

                def read(name: str, root: Path = layout.root) -> bytes:
                    path = root / name
                    if stat.S_ISLNK(os.lstat(path).st_mode):  # check ...
                        raise OSError("link refused")
                    with open(path, "rb") as handle:  # ... then use
                        return handle.read()

                found = module.swap_race(read, layout, iterations=10_000, timeout=5)
                if any("canary" in item.problem for item in found):
                    caught += 1
                elif not found:
                    clean += 1
                else:
                    other += 1
        line = f"vulnerable reader, {label}: caught {caught}/{rounds}, passed clean {clean}, other {other}"
        print(line, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        vulnerable_rates(int(sys.argv[2]))
    else:
        raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 100))
