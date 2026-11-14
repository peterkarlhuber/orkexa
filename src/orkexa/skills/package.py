"""Agent Skills packages (https://agentskills.io/specification): read, check, describe and write them.

Everything here is pure: input is a mapping of relative paths to bytes or the bytes of a ZIP file, output
is models and bytes. Nothing touches the file system, the network or a model, and no script of a skill is
ever run.

The checks follow the specification strictly, with one intended difference from the reference validator
``skills-ref``: an unknown front matter key is a warning, not an error, so skills written for other clients
(for example with ``allowed-tools`` as a list or a ``model`` key) can still be imported. Package files are
kept byte for byte, so :func:`write_zip` after :func:`read_zip` gives the same files back.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import posixpath
import re
import stat
from typing import Literal
import unicodedata
import zipfile

from pydantic import BaseModel, ConfigDict, Field

from ._frontmatter import FrontmatterError, FrontmatterValue, parse_frontmatter, split_frontmatter

__all__ = [
    "KNOWN_KEYS",
    "SKILL_FILE",
    "SkillFile",
    "SkillIssue",
    "SkillLimits",
    "SkillPackage",
    "SkillPackageError",
    "catalog_entry",
    "is_text_path",
    "package_digest",
    "parse_skill",
    "read_zip",
    "render_activation",
    "validate",
    "write_zip",
]

SKILL_FILE = "SKILL.md"
#: The front matter keys of the specification. Others are kept and reported as a warning.
KNOWN_KEYS = frozenset({"name", "description", "license", "compatibility", "metadata", "allowed-tools"})
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DRIVE = re.compile(r"^[A-Za-z]:")
_TEXT_SUFFIXES = (
    ".md",
    ".markdown",
    ".txt",
    ".json",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
)
#: Files a ZIP tool adds on its own; they are dropped on import with a warning.
_JUNK = re.compile(r"(^|/)(__MACOSX/|\.DS_Store$|Thumbs\.db$)")
_DIGEST_PREFIX = b"orkexa-skill/1\n"


class SkillPackageError(ValueError):
    """A package that cannot be read at all (not a ZIP, too large, unsafe paths, no ``SKILL.md``).

    ``code`` is stable (``skill.zip_invalid``, ``skill.too_large``, ``skill.unsafe_path``, ...); ``params``
    holds the values the message names.
    """

    def __init__(self, code: str, message: str, **params: object) -> None:
        super().__init__(message)
        self.code = code
        self.params = params


class SkillLimits(BaseModel):
    """Size limits for one skill. The defaults are those of the Nodestro plan (NAL-B3)."""

    model_config = ConfigDict(frozen=True)

    max_zip_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_total_bytes: int = Field(default=30 * 1024 * 1024, ge=1)
    max_files: int = Field(default=500, ge=1)
    max_depth: int = Field(default=6, ge=1)
    #: The largest ratio of a ZIP entry's size to its compressed size (a higher one is a ZIP bomb).
    max_ratio: int = Field(default=100, ge=1)
    max_skill_md_bytes: int = Field(default=100 * 1024, ge=1)
    #: Above these the body still loads, with a warning (the specification recommends staying below them).
    warn_lines: int = Field(default=500, ge=1)
    warn_tokens: int = Field(default=5000, ge=1)


class SkillIssue(BaseModel):
    """One finding of :func:`validate`. An ``error`` makes a package unusable; a ``warning`` names a risk."""

    model_config = ConfigDict(frozen=True)

    severity: Literal["error", "warning"]
    code: str
    message: str
    path: str | None = None
    field: str | None = None
    params: dict[str, str | int] = Field(default_factory=dict)


class SkillFile(BaseModel):
    """One file of a package, with its exact bytes. ``path`` is relative, with ``/``."""

    model_config = ConfigDict(frozen=True)

    path: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


class SkillPackage(BaseModel):
    """A parsed skill: the front matter fields, the body of ``SKILL.md`` and all files (with ``SKILL.md``).

    ``unknown_frontmatter`` holds keys outside the specification; ``directory`` is the folder name the package
    came in (``None`` for a single ``SKILL.md``). ``frontmatter_error`` is set when the front matter could not
    be read; :func:`validate` reports it.
    """

    model_config = ConfigDict(frozen=True)

    name: str = ""
    description: str = ""
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    allowed_tools: str | None = None
    body: str = ""
    files: tuple[SkillFile, ...] = ()
    unknown_frontmatter: dict[str, FrontmatterValue] = Field(default_factory=dict)
    directory: str | None = None
    frontmatter_error: str | None = None
    #: Problems found while reading (front matter value types), reported by :func:`validate`.
    read_issues: tuple[SkillIssue, ...] = ()

    def file(self, path: str) -> SkillFile | None:
        return next((item for item in self.files if item.path == path), None)

    @property
    def resources(self) -> list[str]:
        """The paths of all files except ``SKILL.md``, sorted."""
        return sorted(item.path for item in self.files if item.path != SKILL_FILE)

    @property
    def has_scripts(self) -> bool:
        return any(item.path.startswith("scripts/") for item in self.files)


def _path_problem(path: str) -> str | None:
    """Why ``path`` is not a safe relative package path, or ``None``."""
    if not path:
        return "empty"
    if "\\" in path:
        return "backslash"
    if any(unicodedata.category(char) in ("Cc", "Cf", "Zl", "Zp") for char in path):
        return "control_character"
    if path.startswith("/") or _DRIVE.match(path):
        return "absolute"
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "dot_segment"
    if unicodedata.normalize("NFC", path) != path:
        return "not_nfc"
    return None


def _check_paths(paths: list[str], limits: SkillLimits) -> None:
    seen: dict[str, str] = {}
    for path in paths:
        problem = _path_problem(path)
        if problem:
            raise SkillPackageError(
                "skill.unsafe_path",
                f"unsafe path in the package: {path!r} ({problem})",
                path=path,
                reason=problem,
            )
        if path.count("/") + 1 > limits.max_depth:
            raise SkillPackageError(
                "skill.unsafe_path",
                f"{path!r} is nested deeper than {limits.max_depth} levels",
                path=path,
                reason="too_deep",
            )
        folded = path.casefold()
        if folded in seen:
            raise SkillPackageError(
                "skill.unsafe_path",
                f"{path!r} and {seen[folded]!r} differ only in case",
                path=path,
                reason="case_clash",
            )
        seen[folded] = path
    if len(paths) > limits.max_files:
        raise SkillPackageError(
            "skill.too_many_files",
            f"the package has more than {limits.max_files} files",
            limit=limits.max_files,
        )


def _single_root(paths: list[str]) -> str | None:
    """The one folder that holds every path and a ``SKILL.md``, if the package is wrapped in one."""
    if SKILL_FILE in paths:
        return None
    roots = {path.split("/", 1)[0] for path in paths}
    if len(roots) == 1 and all("/" in path for path in paths):
        root = next(iter(roots))
        if f"{root}/{SKILL_FILE}" in paths:
            return root
    return None


def parse_skill(files: Mapping[str, bytes], *, limits: SkillLimits | None = None) -> SkillPackage:
    """Parse a package from ``{relative path: bytes}``.

    A package wrapped in one folder (``my-skill/SKILL.md``, ``my-skill/references/...``) is unwrapped and the
    folder name kept as ``directory``. Raises :class:`SkillPackageError` for unsafe paths, too many or too
    large files and a missing ``SKILL.md``; everything else (bad front matter, wrong field values) is left to
    :func:`validate`.
    """
    limits = limits or SkillLimits()
    paths = list(files)
    _check_paths(paths, limits)
    total = sum(len(data) for data in files.values())
    if total > limits.max_total_bytes:
        raise SkillPackageError(
            "skill.too_large",
            f"the package is larger than {limits.max_total_bytes} bytes",
            limit=limits.max_total_bytes,
        )
    directory = _single_root(paths)
    if directory is not None:
        files = {path.split("/", 1)[1]: data for path, data in files.items()}
    if SKILL_FILE not in files:
        raise SkillPackageError("skill.missing_skill_md", "the package has no SKILL.md at its top level")
    raw = files[SKILL_FILE]
    if len(raw) > limits.max_skill_md_bytes:
        raise SkillPackageError(
            "skill.too_large",
            f"SKILL.md is larger than {limits.max_skill_md_bytes} bytes",
            limit=limits.max_skill_md_bytes,
            path=SKILL_FILE,
        )
    ordered = tuple(SkillFile(path=path, data=files[path]) for path in sorted(files))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return SkillPackage(
            files=ordered, directory=directory, frontmatter_error="SKILL.md is not UTF-8 text"
        )
    try:
        head, body = split_frontmatter(text)
        fields = parse_frontmatter(head)
    except FrontmatterError as error:
        return SkillPackage(files=ordered, directory=directory, frontmatter_error=str(error))
    issues: list[SkillIssue] = []
    known: dict[str, object] = {}
    for key in ("name", "description", "license", "compatibility", "allowed-tools"):
        if key not in fields:
            continue
        value = fields[key]
        if key == "allowed-tools" and isinstance(value, list):
            # Some clients write a list; the specification has a space-separated string.
            value = " ".join(value)
        if not isinstance(value, str):
            issues.append(
                SkillIssue(
                    severity="error",
                    code="skill.field_type",
                    field=key,
                    message=f"'{key}' must be a string",
                    path=SKILL_FILE,
                )
            )
            continue
        known[key.replace("-", "_")] = value
    metadata = fields.get("metadata", {})
    if isinstance(metadata, str) and not metadata:
        metadata = {}
    if not isinstance(metadata, dict):
        issues.append(
            SkillIssue(
                severity="error",
                code="skill.field_type",
                field="metadata",
                message="'metadata' must be a mapping of strings",
                path=SKILL_FILE,
            )
        )
        metadata = {}
    return SkillPackage(
        **known,  # type: ignore[arg-type]
        metadata=metadata,
        body=body,
        files=ordered,
        unknown_frontmatter={key: value for key, value in fields.items() if key not in KNOWN_KEYS},
        directory=directory,
        read_issues=tuple(issues),
    )


def read_zip(data: bytes, *, limits: SkillLimits | None = None) -> SkillPackage:
    """Parse a package from the bytes of a ZIP file, with the size, path and ZIP-bomb limits of ``limits``.

    Symbolic links, encrypted entries, duplicate names and entries whose ratio of size to compressed size is
    above ``max_ratio`` are refused. Directory entries are skipped; ``__MACOSX/``, ``.DS_Store`` and
    ``Thumbs.db`` are dropped. Nothing is written to disk.
    """
    limits = limits or SkillLimits()
    if len(data) > limits.max_zip_bytes:
        raise SkillPackageError(
            "skill.too_large",
            f"the ZIP file is larger than {limits.max_zip_bytes} bytes",
            limit=limits.max_zip_bytes,
        )
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, ValueError, OSError) as error:
        raise SkillPackageError("skill.zip_invalid", f"not a readable ZIP file: {error}") from None
    with archive:
        entries = archive.infolist()
        if len(entries) > limits.max_files * 2:
            raise SkillPackageError(
                "skill.too_many_files",
                f"the package has more than {limits.max_files} files",
                limit=limits.max_files,
            )
        files: dict[str, bytes] = {}
        total = 0
        for entry in entries:
            # The stored name: zipfile turns backslashes into "/" only on Windows; a package reads the same
            # everywhere.
            name = entry.orig_filename
            if entry.is_dir() or _JUNK.search(name):
                continue
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise SkillPackageError(
                    "skill.unsafe_path", f"{name!r} is a symbolic link", path=name, reason="symlink"
                )
            if mode and not stat.S_ISREG(mode) and stat.S_IFMT(mode):
                raise SkillPackageError(
                    "skill.unsafe_path", f"{name!r} is not a regular file", path=name, reason="special_file"
                )
            if entry.flag_bits & 0x1:
                raise SkillPackageError("skill.zip_invalid", f"{name!r} is encrypted", path=name)
            if name in files:
                raise SkillPackageError(
                    "skill.unsafe_path", f"{name!r} is in the ZIP file twice", path=name, reason="duplicate"
                )
            problem = _path_problem(name)
            if problem:
                raise SkillPackageError(
                    "skill.unsafe_path",
                    f"unsafe path in the package: {name!r} ({problem})",
                    path=name,
                    reason=problem,
                )
            if entry.file_size > max(entry.compress_size, 1) * limits.max_ratio and entry.file_size > 1024:
                raise SkillPackageError(
                    "skill.zip_bomb",
                    f"{name!r} expands more than {limits.max_ratio} times",
                    path=name,
                    limit=limits.max_ratio,
                )
            if total + entry.file_size > limits.max_total_bytes:
                raise SkillPackageError(
                    "skill.too_large",
                    f"the unpacked package is larger than {limits.max_total_bytes} bytes",
                    limit=limits.max_total_bytes,
                )
            try:
                with archive.open(entry) as handle:
                    # One byte more than declared: a lying header cannot make us read past the limit.
                    content = handle.read(entry.file_size + 1)
            except (zipfile.BadZipFile, NotImplementedError, RuntimeError, OSError, EOFError) as error:
                raise SkillPackageError(
                    "skill.zip_invalid", f"{name!r} cannot be read: {error}", path=name
                ) from None
            if len(content) != entry.file_size:
                raise SkillPackageError(
                    "skill.zip_invalid", f"{name!r} does not match its declared size", path=name
                )
            total += len(content)
            files[name] = content
    return parse_skill(files, limits=limits)


def is_text_path(path: str) -> bool:
    """Whether a package file is text a model may read (by its suffix)."""
    return path.lower().endswith(_TEXT_SUFFIXES)


def validate(package: SkillPackage, *, limits: SkillLimits | None = None) -> list[SkillIssue]:
    """All findings for ``package``: errors first, then warnings, each in file order."""
    limits = limits or SkillLimits()
    issues: list[SkillIssue] = []

    def error(code: str, message: str, **extra: object) -> None:
        issues.append(SkillIssue(severity="error", code=code, message=message, path=SKILL_FILE, **extra))  # type: ignore[arg-type]

    def warning(code: str, message: str, **extra: object) -> None:
        issues.append(SkillIssue(severity="warning", code=code, message=message, **extra))  # type: ignore[arg-type]

    if package.frontmatter_error:
        error("skill.frontmatter_invalid", f"the front matter cannot be read: {package.frontmatter_error}")
        return issues
    issues.extend(package.read_issues)
    name = package.name
    if not name:
        error("skill.name_missing", "'name' is required", field="name")
    elif len(name) > 64 or not _NAME.match(name):
        error(
            "skill.name_invalid",
            "'name' must be 1-64 lowercase letters, digits and single hyphens, "
            "not starting or ending with one",
            field="name",
            params={"name": name[:80]},
        )
    elif package.directory is not None and package.directory != name:
        error(
            "skill.name_mismatch",
            f"'name' must equal the folder name {package.directory!r}",
            field="name",
            params={"name": name, "directory": package.directory},
        )
    description = package.description.strip()
    if not description:
        error("skill.description_missing", "'description' is required", field="description")
    elif len(package.description) > 1024:
        error(
            "skill.description_too_long",
            "'description' must be at most 1024 characters",
            field="description",
            params={"limit": 1024},
        )
    if package.compatibility is not None and not 1 <= len(package.compatibility) <= 500:
        error(
            "skill.compatibility_invalid",
            "'compatibility' must be 1-500 characters",
            field="compatibility",
            params={"limit": 500},
        )
    for key in package.unknown_frontmatter:
        warning(
            "skill.unknown_key",
            f"'{key}' is not part of the Agent Skills specification; it is kept but unused",
            path=SKILL_FILE,
            field=key,
        )
    if package.allowed_tools:
        warning(
            "skill.allowed_tools_ignored",
            "'allowed-tools' is shown but never grants tools; tools come only from the agent's settings",
            path=SKILL_FILE,
            field="allowed-tools",
        )
    lines = package.body.count("\n") + 1
    if lines > limits.warn_lines:
        warning(
            "skill.body_long",
            f"the instructions have {lines} lines; the specification recommends at most {limits.warn_lines}",
            path=SKILL_FILE,
            params={"lines": lines, "limit": limits.warn_lines},
        )
    tokens = len(package.body) // 4
    if tokens > limits.warn_tokens:
        warning(
            "skill.body_large",
            f"the instructions have about {tokens} tokens; the specification recommends at "
            f"most {limits.warn_tokens}",
            path=SKILL_FILE,
            params={"tokens": tokens, "limit": limits.warn_tokens},
        )
    if package.has_scripts:
        warning(
            "skill.scripts_not_run",
            "the package contains scripts; they are kept but never run",
            path="scripts/",
        )
    for item in package.files:
        if item.path != SKILL_FILE and is_text_path(item.path):
            try:
                item.data.decode("utf-8")
            except UnicodeDecodeError:
                warning(
                    "skill.file_not_utf8",
                    f"{item.path} is not UTF-8 text and cannot be read by a model",
                    path=item.path,
                )
    return sorted(issues, key=lambda issue: issue.severity != "error")


def package_digest(package: SkillPackage) -> str:
    """``sha256:<hex>`` over every path and its bytes, in sorted order: equal files give an equal digest."""
    digest = hashlib.sha256(_DIGEST_PREFIX)
    for item in sorted(package.files, key=lambda file: file.path):
        path = item.path.encode("utf-8")
        digest.update(len(path).to_bytes(4, "big") + path + len(item.data).to_bytes(8, "big") + item.data)
    return "sha256:" + digest.hexdigest()


def catalog_entry(package: SkillPackage) -> dict[str, str]:
    """The first loading stage: name and description, which an agent sees for every assigned skill."""
    return {"name": package.name, "description": package.description.strip()}


def _escape_attribute(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def render_activation(package: SkillPackage) -> str:
    """The second loading stage: the instructions in a ``<skill_content>`` wrapper and the list of resources.

    Resources are listed by path only; their content is the third stage and read on demand.
    """
    resources = package.resources
    listing = "".join(f"\n  <file>{_escape_attribute(path)}</file>" for path in resources)
    files = f"\n<skill_resources>{listing}\n</skill_resources>" if resources else ""
    head = f'<skill_content name="{_escape_attribute(package.name)}">'
    return f"{head}\n{package.body.strip()}\n</skill_content>{files}"


def write_zip(package: SkillPackage) -> bytes:
    """A standard ZIP of the package in a folder named after the skill, with the files' exact bytes.

    The output is reproducible: sorted entries, a fixed timestamp, mode ``0644`` for every file (scripts
    are not marked executable).
    """
    folder = package.name or package.directory or "skill"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(package.files, key=lambda file: file.path):
            info = zipfile.ZipInfo(posixpath.join(folder, item.path), date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, item.data)
    return buffer.getvalue()
