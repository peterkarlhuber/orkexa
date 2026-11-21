"""A reference implementation of the folder grant contract (T-CONF-9) on :class:`orkexa.fs.SafeRoot`.

It is the smallest grant store that passes :data:`~orkexa.testing.conformance.FOLDER_CASES`: grants in
memory, a new :class:`SafeRoot` per call with the identity recorded at the grant, the data folder protected,
and an access log of ``(operation, outcome)``. Nodestro's folder grants (nodestro-runtime ``folders``) run
the same cases.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from orkexa.fs import PDF_SUFFIXES, FsError, SafeRoot
from orkexa.testing.conformance import DATA_FOLDER, Request


@dataclass
class _Grant:
    path: str
    identity: tuple[int, int]
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    max_file_bytes: int | None
    expires_at: datetime | None
    revoked: bool = False
    log: list[tuple[str, str]] = field(default_factory=list)


class SafeRootFolders:
    """Folder grants on :class:`SafeRoot` (see the module documentation)."""

    def __init__(self, data: Path) -> None:
        data.mkdir(exist_ok=True)
        self.data = data
        self.grants: dict[str, _Grant] = {}

    def grant(
        self,
        path: str,
        *,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        max_file_bytes: int | None = None,
        expires_at: datetime | None = None,
    ) -> str:
        root = SafeRoot(path, protected=[self.data], include=include, exclude=exclude)
        key = f"g{len(self.grants) + 1}"
        self.grants[key] = _Grant(
            root.path, root.identity, tuple(include), tuple(exclude), max_file_bytes, expires_at
        )
        return key

    def revoke(self, grant: str) -> None:
        self.grants[grant].revoked = True

    def _root(self, grant: _Grant) -> SafeRoot:
        if grant.revoked or (grant.expires_at is not None and grant.expires_at <= datetime.now(UTC)):
            raise FsError("fs.revoked", "The grant was revoked or has expired")
        kwargs = {} if grant.max_file_bytes is None else {"max_file_bytes": grant.max_file_bytes}
        return SafeRoot(
            grant.path,
            protected=[self.data],
            include=grant.include,
            exclude=grant.exclude,
            expect_identity=grant.identity,
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

    def list(self, grant: str, path: str = "") -> list[str]:
        return self._logged(grant, "list", lambda root: [entry.path for entry in root.list(path)])  # type: ignore[return-value]

    def read(self, grant: str, path: str | bytes) -> str:
        return self._logged(grant, "read", lambda root: _text(root, path))  # type: ignore[return-value]

    def search(self, grant: str, text: str) -> list[str]:
        return self._logged(grant, "search", lambda root: _search(root, text.casefold()))  # type: ignore[return-value]

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


def _search(root: SafeRoot, needle: str, folder: str = "") -> list[str]:
    found: list[str] = []
    for entry in root.list(folder):
        if entry.kind == "dir":
            found += _search(root, needle, entry.path)
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
