"""Shipped Dockerfiles pin their base image by digest and install Python packages only by hash (DS-9)."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SKIPPED = {".git", ".venv", "node_modules", "build", "dist"}


def _is_dockerfile(path: Path) -> bool:
    name = path.name
    return name == "Dockerfile" or name.endswith(".Dockerfile") or name.startswith("Dockerfile.")


DOCKERFILES = sorted(
    path
    for path in ROOT.rglob("*")
    if path.is_file() and _is_dockerfile(path) and not _SKIPPED & set(path.relative_to(ROOT).parts)
)
_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
_REQUIREMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^\s\\]+$")


def _instructions(path: Path) -> list[str]:
    """The Dockerfile's instructions, with continuation lines joined and comments left out."""
    joined = re.sub(r"\\\n\s*", " ", path.read_text(encoding="utf-8"))
    lines = (line.strip() for line in joined.splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def test_the_sandbox_dockerfile_is_found() -> None:
    assert ROOT / "docker" / "sandbox.Dockerfile" in DOCKERFILES


def unpinned(path: Path) -> list[str]:
    """The base images of ``path`` that are not pinned by digest (earlier stages and ``scratch`` aside)."""
    stages: set[str] = set()
    found = []
    for line in _instructions(path):
        words = line.split()
        if words[0].upper() != "FROM":
            continue
        args = [word for word in words[1:] if not word.startswith("--")]
        image = args[0]
        if len(args) >= 3 and args[1].upper() == "AS":
            stages.add(args[2].lower())
        if image.lower() not in stages and image != "scratch" and not _DIGEST.search(image):
            found.append(image)
    return found


def unhashed_installs(path: Path) -> list[str]:
    """The ``pip install`` commands of ``path`` that do not install a requirements file by hash."""
    return [
        " ".join(command.split())
        for line in _instructions(path)
        for command in re.split(r"&&|;|\|\|", line)
        if re.search(r"\bpip3?\s+install\b", command)
        and not ("--require-hashes" in command and re.search(r"\s-r\s", command))
    ]


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_base_images_are_pinned_by_digest(path: Path) -> None:
    assert unpinned(path) == [], f"{path.name}: pin FROM by digest (image@sha256:...)"


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_pip_installs_only_hashed_requirements(path: Path) -> None:
    assert unhashed_installs(path) == []


def test_the_checks_refuse_what_they_should(tmp_path: Path) -> None:
    digest = "@sha256:" + "a" * 64
    dockerfile = tmp_path / "x.Dockerfile"
    dockerfile.write_text(
        "# FROM ignored:latest\n"
        f"FROM --platform=linux/amd64 python:3.12-slim{digest} AS build\n"
        "FROM build\nFROM scratch\nFROM python:3.12-slim\nFROM ghcr.io/acme/base@sha256:short\n"
        "RUN pip install --require-hashes -r r.txt && pip install \\\n    requests\n"
    )
    assert unpinned(dockerfile) == ["python:3.12-slim", "ghcr.io/acme/base@sha256:short"]
    assert unhashed_installs(dockerfile) == ["pip install requests"]


def test_the_sandbox_requirements_are_complete_and_hashed() -> None:
    text = (ROOT / "docker" / "sandbox-requirements.txt").read_text(encoding="utf-8")
    requirements = re.sub(r"\\\n\s*", " ", text)
    lines = requirements.splitlines()
    entries = [line.split() for line in lines if line.strip() and not line.startswith("#")]
    names = set()
    for entry in entries:
        assert _REQUIREMENT.match(entry[0]), entry[0]
        hashes = [word for word in entry[1:] if word.startswith("--hash=sha256:")]
        assert hashes and all(re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", word) for word in hashes), entry
        names.add(entry[0].split("==")[0].lower())
    # orkexa's runtime dependencies (pyproject.toml) and what they need
    assert {"pydantic", "pydantic-core", "typing-extensions", "annotated-types", "typing-inspection"} <= names
