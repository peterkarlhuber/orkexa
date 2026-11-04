"""ORK3-01: links in the runtime archive that leave the runtime folder through other links.

Internal backend audit round 3 (2026-11-04). No network.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import tarfile

import pytest

from orkexa.local import OllamaError, install_runtime
from orkexa.local.install import RuntimeArtifact

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX links and process groups")


# ---- ORK3-01: link chains in the runtime archive ------------------------------------------


def _archive(path: Path, links: list[tuple[str, str]]) -> RuntimeArtifact:
    with tarfile.open(path, "w:gz") as bundle:
        data = b"#!/bin/sh\n"
        info = tarfile.TarInfo("bin/ollama")
        info.size, info.mode = len(data), 0o755
        bundle.addfile(info, io.BytesIO(data))
        for name, target in links:
            link = tarfile.TarInfo(name)
            link.type, link.linkname = tarfile.SYMTYPE, target
            bundle.addfile(link)
    return RuntimeArtifact("darwin", "any", path.name, hashlib.sha256(path.read_bytes()).hexdigest())


@posix_only
@pytest.mark.parametrize(
    "links",
    [
        # each link looks inside on its own; together `escape` points two levels above the runtime
        [("x", "."), ("lib/escape", "../x/x/../../outside")],
        # the escaping link comes first; a later link is created through it
        [("a", "x/.."), ("x", "."), ("a/planted", "anything")],
    ],
)
def test_install_refuses_link_chains_that_leave_the_runtime(
    tmp_path: Path, links: list[tuple[str, str]]
) -> None:
    home = tmp_path / "home"
    archive = tmp_path / "ollama-darwin.tgz"
    artifact = _archive(archive, links)
    with pytest.raises(OllamaError) as error:
        install_runtime(home, archive=archive, artifact=artifact, gpu=False)
    assert error.value.code == "unsafe_archive"
    assert not (home / "runtime" / "0.40.14").exists()
    planted = [p for p in tmp_path.rglob("planted")]
    assert planted == []  # nothing was created outside the scratch folder either


@posix_only
def test_install_keeps_links_inside_the_runtime(tmp_path: Path) -> None:
    archive = tmp_path / "ollama-darwin.tgz"
    artifact = _archive(archive, [("lib/ollama/libx.so", "libx.so.1"), ("lib/current", "ollama")])
    binary = install_runtime(tmp_path / "home", archive=archive, artifact=artifact, gpu=False)
    assert (binary.parent.parent / "lib/current").is_symlink()
