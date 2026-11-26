"""TEMPORARY (W3-S, removed before handover): repeat the conformance swap race and count its outcomes.

Runs the ``folders.swap_race`` case's two races (a file, a folder) ``N`` times with the case's own parameters
against the reference folder access of the conformance tests, and prints how often each finding occurred, how many
reads returned the regular file, and why the others were refused.
"""

from __future__ import annotations

from collections import Counter
from functools import partial
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "conformance"))

from folder_access import SafeRootFolders  # noqa: E402

from orkexa.testing import pathguard  # noqa: E402


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

                def read(path: str, seen: Counter[str] = seen) -> bytes:
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


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 100))
