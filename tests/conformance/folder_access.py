"""A reference implementation of the folder grant contract (T-CONF-9) on :class:`orkexa.fs.SafeRoot`.

It is the smallest grant store that passes :data:`~orkexa.testing.conformance.FOLDER_CASES`: grants in
memory, a new :class:`SafeRoot` per call with the identity recorded at the grant, the data folder protected,
denies checked on every entry (canonical, compared without case), the grant's agents and mode, and an access
log of ``(operation, outcome)``. Nodestro's folder grants (nodestro-runtime ``folders``) run
the same cases.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import os
from pathlib import Path

from orkexa.fs import PDF_SUFFIXES, FsError, SafeRoot, fold
from orkexa.testing.conformance import DATA_FOLDER, Request


@dataclass
class _Grant:
    path: str
    identity: tuple[int, int]
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    max_file_bytes: int | None
    expires_at: datetime | None
    agents: tuple[str, ...] = ()
    unattended: bool = False
    revoked: bool = False
    log: list[tuple[str, str]] = field(default_factory=list)


class SafeRootFolders:
    """Folder grants on :class:`SafeRoot` (see the module documentation)."""

    def __init__(self, data: Path, *, agent: str | None = None, unattended: bool = False) -> None:
        data.mkdir(exist_ok=True)
        self.data = data
        self.grants: dict[str, _Grant] = {}
        self.denies: list[tuple[str, ...]] = []
        self.agent = agent
        self.unattended = unattended

    def as_caller(self, agent: str, *, unattended: bool = False) -> SafeRootFolders:
        found = SafeRootFolders(self.data, agent=agent, unattended=unattended)
        found.grants, found.denies = self.grants, self.denies  # the same grants and denies
        return found

    def deny(self, path: str) -> None:
        self.denies.append(_parts(path))

    def _denied(self, target: str) -> bool:
        # ``target`` lies below a canonical root (the grant's path and names below it): no file-system
        # access here, which would slow every open and disturb the swap race.
        parts = _folded(target)
        return any(parts[: len(deny)] == deny for deny in self.denies)

    def grant(
        self,
        path: str,
        *,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        max_file_bytes: int | None = None,
        expires_at: datetime | None = None,
        agents: Sequence[str] = (),
        unattended: bool = False,
    ) -> str:
        root = SafeRoot(path, protected=[self.data], include=include, exclude=exclude)
        key = f"g{len(self.grants) + 1}"
        self.grants[key] = _Grant(
            root.path,
            root.identity,
            tuple(include),
            tuple(exclude),
            max_file_bytes,
            expires_at,
            tuple(agents),
            unattended,
        )
        return key

    def revoke(self, grant: str) -> None:
        self.grants[grant].revoked = True

    def _root(self, grant: _Grant) -> SafeRoot:
        if grant.revoked or (grant.expires_at is not None and grant.expires_at <= datetime.now(UTC)):
            raise FsError("fs.revoked", "The grant was revoked or has expired")
        if (grant.agents and self.agent not in grant.agents) or (self.unattended and not grant.unattended):
            raise FsError("fs.not_granted", "The grant does not cover this agent or mode")
        if self._denied(grant.path):
            raise FsError("fs.denied", "A person denied this folder")
        kwargs = {} if grant.max_file_bytes is None else {"max_file_bytes": grant.max_file_bytes}
        return SafeRoot(
            grant.path,
            protected=[self.data],
            include=grant.include,
            exclude=grant.exclude,
            expect_identity=grant.identity,
            on_open=self._opened(grant.path),
            **kwargs,  # type: ignore[arg-type]
        )

    def _logged(self, grant: str, operation: str, call: object) -> object:
        found = self.grants[grant]
        try:
            result = call(self._root(found))  # type: ignore[operator]
        except FsError:
            found.log.append((operation, "denied"))
            raise
        except Exception:
            found.log.append((operation, "failure"))
            raise
        found.log.append((operation, "success"))
        return result

    def _opened(self, path: str) -> Callable[[tuple[str, ...]], None]:
        """Checks the denies against the names the file system stored for what is opened."""

        def check(names: tuple[str, ...]) -> None:
            if self._denied(os.path.join(path, *names)):
                raise FsError("fs.denied", "A person denied this path")

        return check

    def _allowed(self, root: SafeRoot) -> Callable[[str], bool]:
        return lambda relative: not self._denied(os.path.join(root.path, *relative.split("/")))

    def _list(self, root: SafeRoot, path: str) -> list[str]:
        if not self._allowed(root)("/".join(root.check(path, kind="dir"))):
            raise FsError("fs.denied", "A person denied this folder")
        return [entry.path for entry in root.list(path) if self._allowed(root)(entry.path)]

    def _read(self, root: SafeRoot, path: str | bytes) -> str:
        if not self._allowed(root)("/".join(root.check(path, kind="file"))):
            raise FsError("fs.denied", "A person denied this file")
        return _text(root, path)

    def list(self, grant: str, path: str = "") -> list[str]:
        return self._logged(grant, "list", lambda root: self._list(root, path))  # type: ignore[return-value]

    def read(self, grant: str, path: str | bytes) -> str:
        return self._logged(grant, "read", lambda root: self._read(root, path))  # type: ignore[return-value]

    def search(self, grant: str, text: str) -> list[str]:
        return self._logged(  # type: ignore[return-value]
            grant, "search", lambda root: _search(root, text.casefold(), allowed=self._allowed(root))
        )

    def accesses(self, grant: str) -> list[tuple[str, str]]:
        return list(self.grants[grant].log)


def _text(root: SafeRoot, path: str | bytes) -> str:
    shown = path if isinstance(path, str) else path.decode("utf-8", "replace")
    if shown.casefold().endswith(tuple(PDF_SUFFIXES)):
        return root.extract_text(path).text
    data = root.read_bytes(path)
    if b"\x00" in data[:8192]:
        raise FsError("fs.unsupported_type", "The file is binary, not text")
    return data.decode("utf-8-sig", errors="replace")


def _parts(path: str) -> tuple[str, ...]:
    """The canonical components of an absolute path a person denied, compared as the guard compares names."""
    return _folded(os.path.realpath(path))


def _folded(path: str) -> tuple[str, ...]:
    return tuple(fold(part) for part in os.path.normpath(path).replace("\\", "/").split("/"))


def _search(
    root: SafeRoot, needle: str, folder: str = "", *, allowed: Callable[[str], bool] = lambda path: True
) -> list[str]:
    found: list[str] = []
    for entry in root.list(folder):
        if not allowed(entry.path):
            continue
        if entry.kind == "dir":
            found += _search(root, needle, entry.path, allowed=allowed)
        elif entry.kind == "file":
            if needle in entry.name.casefold():
                found.append(entry.path)
                continue
            try:
                text = _text(root, entry.path)
            except FsError:
                continue
            if needle in text.casefold():
                found.append(entry.path)
    return found


def safe_root_folders(request: Request) -> SafeRootFolders:
    return SafeRootFolders(request.workdir / DATA_FOLDER)
