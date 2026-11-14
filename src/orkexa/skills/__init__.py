"""Agent Skills (https://agentskills.io): read, check and write skill packages (ADR 0018).

* :func:`read_zip` and :func:`parse_skill` read a package (ZIP bytes or ``{path: bytes}``) into a
  :class:`SkillPackage`, refusing unsafe paths, symbolic links and ZIP bombs (:class:`SkillPackageError`).
* :func:`validate` lists the :class:`SkillIssue` findings: the specification's rules as errors, unknown
  front matter keys, ``allowed-tools``, long instructions and scripts as warnings.
* :func:`catalog_entry` and :func:`render_activation` give the first two loading stages for a model;
  :func:`package_digest` and :func:`write_zip` identify and export a package byte for byte.

Only pydantic and the standard library are needed. Nothing here touches the disk or the network, and
scripts in a package are never run.
"""

from __future__ import annotations

from ._frontmatter import FrontmatterError, parse_frontmatter, split_frontmatter
from .package import (
    KNOWN_KEYS,
    SKILL_FILE,
    SkillFile,
    SkillIssue,
    SkillLimits,
    SkillPackage,
    SkillPackageError,
    catalog_entry,
    is_text_path,
    package_digest,
    parse_skill,
    read_zip,
    render_activation,
    validate,
    write_zip,
)

__all__ = [
    "KNOWN_KEYS",
    "SKILL_FILE",
    "FrontmatterError",
    "SkillFile",
    "SkillIssue",
    "SkillLimits",
    "SkillPackage",
    "SkillPackageError",
    "catalog_entry",
    "is_text_path",
    "package_digest",
    "parse_frontmatter",
    "parse_skill",
    "read_zip",
    "render_activation",
    "split_frontmatter",
    "validate",
    "write_zip",
]
