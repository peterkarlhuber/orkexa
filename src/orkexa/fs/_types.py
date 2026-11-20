"""Small value types shared by the platform modules of :mod:`orkexa.fs`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Info:
    """What ``fstat`` (POSIX) or ``GetFileInformationByHandle`` (Windows) says about an open file or folder.

    ``kind`` is ``"dir"``, ``"file"`` or ``"other"`` (FIFO, socket, device). ``dev`` and ``ino`` are the
    identity: ``st_dev`` and ``st_ino`` on POSIX, the volume serial number and the file index on Windows.
    """

    kind: str
    dev: int
    ino: int
    size: int
    nlink: int
    mtime_ns: int
