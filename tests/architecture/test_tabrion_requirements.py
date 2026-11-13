"""TABRION VER-1 (V-B0, RC-1, RC-8): ``requirements/tabrion.txt`` and the Tabrion requirements of the
manifest.

Orkexa is a library: ``pyproject.toml`` names Tabrion by a range of final versions only
(``tabrion[...]>=0.1,<0.2``), never by a direct URL (RC-1) and never with a pre-release specifier (RC-8).
Without a package index such a range cannot be resolved, so ``requirements/tabrion.txt`` pins the one
Tabrion that a standalone install and the CI use (``pip install -r requirements/tabrion.txt -e ".[dev]"``,
pip 25.0 or newer; plan doc 06 section 6.4).

The file holds exactly one requirement line in the canonical form
``tabrion[<extras>] @ git+https://github.com/peterkarlhuber/tabrion.git@<40-hex SHA>`` (extras optional),
and the SHA is the commit of a final Tabrion version listed in :data:`FINAL_TABRION_COMMITS`. A new pin adds
its SHA there in the same change, which the reviewer checks against Tabrion's history. The pinned version
satisfies every Tabrion range of the manifest. The probes below turn each rule red.

TABRION DC-OR-1 (dependency confusion): the installed Tabrion must be that pin, not merely a version inside
the range. Its ``direct_url.json`` must name ``https://github.com/peterkarlhuber/tabrion.git`` at the pinned
commit; a Tabrion from a package index (no ``direct_url.json``), a local checkout or another commit fails.
Every CI job that runs the suite installs with ``-r requirements/tabrion.txt``, so every one of them checks
this (before, only the ``tabrion-install`` job did, inline). That job's ``poisoned index`` leg also runs the
install with ``scripts/check_tabrion_source.py``: a poisoned index must get no request for ``tabrion`` at
all.
"""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import re
import tomllib
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
import pytest

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "requirements" / "tabrion.txt"
PYPROJECT = ROOT / "pyproject.toml"

#: Commits of final Tabrion versions (``src/tabrion/_version.py`` at that commit). v0.1.0 = TABRION TAB-040.
FINAL_TABRION_COMMITS = {"d5821faa8742dea012abda3b663549c038d76543": "0.1.0"}

PIN = re.compile(
    r"tabrion(?:\[[a-z0-9_-]+(?:,[a-z0-9_-]+)*\])?"
    r" @ git\+https://github\.com/peterkarlhuber/tabrion\.git@(?P<sha>[0-9a-f]{40})"
)


def requirement_lines(text: str) -> list[str]:
    """The lines of a requirements file without comments and blank lines."""
    lines = []
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].strip() if not raw.lstrip().startswith("#") else ""
        if line:
            lines.append(line)
    return lines


def manifest_requirements(pyproject: dict[str, Any]) -> list[tuple[str, str]]:
    """``(where, requirement)`` for every requirement of the core, the extras and the dependency groups."""
    project = pyproject.get("project", {})
    found = [("dependencies", item) for item in project.get("dependencies", [])]
    for extra, items in project.get("optional-dependencies", {}).items():
        found.extend((f"optional-dependencies.{extra}", item) for item in items)
    for group, items in pyproject.get("dependency-groups", {}).items():
        found.extend((f"dependency-groups.{group}", item) for item in items if isinstance(item, str))
    return found


def range_problem(requirement: Requirement) -> str | None:
    """Why a Tabrion requirement is not a range of one minor line of final versions, or ``None``.

    The only accepted shape is ``>=A.B[.C],<A.(B+1)`` for ``A == 0`` (RC-3: a 0.x minor may break) and
    ``>=A.B[.C],<(A+1)`` from 1.0 on (RC-1, RC-8): no pre-release, no other operator, no second line.
    """
    versions: dict[str, Version] = {}
    for spec in requirement.specifier:
        try:
            version = Version(spec.version)
        except InvalidVersion:
            return f"has an unparseable specifier {str(spec)!r}; use a range like >=0.1,<0.2"
        if version.is_prerelease or version.is_postrelease or version.local:
            return f"has a pre-release or non-final specifier {str(spec)!r} (RC-8)"
        if spec.operator not in (">=", "<") or spec.operator in versions:
            return "is not a range like >=0.1,<0.2 of one minor line (RC-1, RC-8)"
        versions[spec.operator] = version
    low, high = versions.get(">="), versions.get("<")
    if low is None or high is None:
        return "is not a range like >=0.1,<0.2 of one minor line (RC-1, RC-8)"
    expected = f"{low.major}.{low.minor + 1}" if low.major == 0 else f"{low.major + 1}"
    if str(high) != expected or len(low.release) > 3:
        return f"is not a range of one minor line: the upper bound must be <{expected} (RC-1, RC-8)"
    return None


def check(requirements_text: str | None, pyproject: dict[str, Any]) -> list[str]:
    """Every rule violation of the pin file and the manifest (empty when both are fine)."""
    errors: list[str] = []
    ranges: list[tuple[str, Requirement]] = []
    for where, item in manifest_requirements(pyproject):
        try:
            requirement = Requirement(item)
        except InvalidRequirement as exc:
            errors.append(f"{where}: invalid requirement {item!r}: {exc}")
            continue
        if canonicalize_name(requirement.name) != "tabrion":
            continue
        if requirement.url is not None:
            errors.append(f"{where}: {item!r} names Tabrion by a direct URL (RC-1); use a range (>=0.1,<0.2)")
            continue
        problem = range_problem(requirement)
        if problem is not None:
            errors.append(f"{where}: {item!r} {problem}")
        ranges.append((where, requirement))
    if requirements_text is None:
        if ranges:
            errors.append("requirements/tabrion.txt is missing, but pyproject.toml names tabrion")
        return errors
    lines = requirement_lines(requirements_text)
    if len(lines) != 1:
        errors.append(f"requirements/tabrion.txt must hold exactly one requirement line, not {len(lines)}")
        return errors
    match = PIN.fullmatch(lines[0])
    if match is None:
        errors.append(
            f"requirements/tabrion.txt: {lines[0]!r} is not 'tabrion[<extras>] @ "
            "git+https://github.com/peterkarlhuber/tabrion.git@<40-hex SHA>'"
        )
        return errors
    version = FINAL_TABRION_COMMITS.get(match["sha"])
    if version is None:
        errors.append(f"requirements/tabrion.txt: {match['sha']} is not a final Tabrion commit")
        return errors
    for where, requirement in ranges:
        if not requirement.specifier.contains(version):
            errors.append(f"{where}: {requirement} does not contain the pinned Tabrion {version}")
    return errors


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_the_pin_and_the_manifest_follow_the_rules() -> None:
    assert check(REQUIREMENTS.read_text(encoding="utf-8"), _pyproject()) == []


def test_the_final_commits_are_final_versions() -> None:
    for sha, version in FINAL_TABRION_COMMITS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", sha)
        parsed = Version(version)
        assert not (parsed.is_prerelease or parsed.is_postrelease or parsed.local), version


SHA = "d5821faa8742dea012abda3b663549c038d76543"
GOOD = f"tabrion @ git+https://github.com/peterkarlhuber/tabrion.git@{SHA}\n"


def _with(extra: str, items: list[str]) -> dict[str, Any]:
    pyproject = _pyproject()
    pyproject["project"].setdefault("optional-dependencies", {})[extra] = items
    return pyproject


def test_extras_on_the_pin_are_allowed() -> None:
    pinned = f"tabrion[testing,postgres] @ git+https://github.com/peterkarlhuber/tabrion.git@{SHA}"
    assert check(pinned, _with("probe", ["tabrion[testing,postgres]>=0.1,<0.2"])) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param(GOOD + GOOD, "exactly one requirement line, not 2", id="two-pins"),
        pytest.param("# nothing\n", "exactly one requirement line, not 0", id="no-pin"),
        pytest.param(GOOD.replace(SHA, SHA[:12]), "is not 'tabrion", id="short-sha"),
        pytest.param(GOOD.replace(SHA, "v0.1.0"), "is not 'tabrion", id="tag-ref"),
        pytest.param(GOOD.replace("tabrion.git@", "tabrion@"), "is not 'tabrion", id="no-dot-git"),
        pytest.param(GOOD.replace("git+https", "git+http"), "is not 'tabrion", id="http"),
        pytest.param(GOOD.replace("peterkarlhuber", "someone"), "is not 'tabrion", id="other-repository"),
        pytest.param("tabrion==0.1.0\n", "is not 'tabrion", id="version-pin"),
        pytest.param("tabrion>=0.1.0a1,<0.2\n", "is not 'tabrion", id="pre-release-range"),
        pytest.param(
            GOOD.replace(SHA, "85145cb" + SHA[7:]), "is not a final Tabrion commit", id="non-final-sha"
        ),
    ],
)
def test_a_bad_pin_file_turns_red(text: str, message: str) -> None:
    errors = check(text, _pyproject())
    assert any(message in error for error in errors), errors


@pytest.mark.parametrize(
    ("items", "message"),
    [
        pytest.param(
            [f"tabrion[postgres] @ git+https://github.com/peterkarlhuber/tabrion.git@{SHA}"],
            "direct URL (RC-1)",
            id="direct-url",
        ),
        pytest.param(["tabrion[postgres]>=0.1.0a1,<0.2"], "pre-release or non-final", id="pre-release"),
        pytest.param(["tabrion[postgres]>=0.1.0rc1,<0.2"], "pre-release or non-final", id="release-cand"),
        pytest.param(["tabrion==0.1.0.dev0"], "pre-release or non-final", id="dev-pin"),
        pytest.param(["tabrion>=0.1,<0.2.0.post1"], "pre-release or non-final", id="post-release"),
        pytest.param(["tabrion>=0.1"], "not a range like", id="unbounded"),
        pytest.param(["tabrion<0.2"], "not a range like", id="no-lower-bound"),
        pytest.param(["tabrion"], "not a range like", id="bare-name"),
        pytest.param(["tabrion==0.1.0"], "not a range like", id="exact-pin"),
        pytest.param(["tabrion~=0.1"], "not a range like", id="compatible-release"),
        pytest.param(["tabrion>0.1,<0.2"], "not a range like", id="exclusive-lower"),
        pytest.param(["tabrion>=0.1,<=0.2"], "not a range like", id="inclusive-upper"),
        pytest.param(["tabrion>=0.1,<0.2,!=0.1.1"], "not a range like", id="exclusion"),
        pytest.param(["tabrion>=0.1,<0.3"], "upper bound must be <0.2", id="two-minor-lines"),
        pytest.param(["tabrion>=0.1,<1"], "upper bound must be <0.2", id="up-to-one"),
        pytest.param(["tabrion>=0.1,<0.2.5"], "upper bound must be <0.2", id="patch-upper"),
        pytest.param(["tabrion===0.1.0-foo"], "unparseable specifier", id="arbitrary-equality"),
        pytest.param(["tabrion>=0.2,<0.3"], "does not contain the pinned Tabrion 0.1.0", id="outside-range"),
    ],
)
def test_a_bad_manifest_requirement_turns_red(items: list[str], message: str) -> None:
    errors = check(GOOD, _with("probe", items))
    assert any(message in error for error in errors), errors


def test_a_manifest_range_needs_the_pin_file() -> None:
    errors = check(None, _with("probe", ["tabrion[postgres]>=0.1,<0.2"]))
    assert errors == ["requirements/tabrion.txt is missing, but pyproject.toml names tabrion"]


@pytest.mark.parametrize("item", ["tabrion>=0.1,<0.2", "tabrion[postgres]>=0.1.3,<0.2", "tabrion>=1.2,<2"])
def test_a_range_of_one_minor_line_is_accepted(item: str) -> None:
    assert range_problem(Requirement(item)) is None


def installed_problems(version: str, direct_url: dict[str, Any] | None, pin_sha: str) -> list[str]:
    """Why the installed Tabrion (``version``, its ``direct_url.json``) is not the pin at ``pin_sha``."""
    if not direct_url:
        return [f"tabrion {version} has no direct_url.json: installed from a package index, not from the pin"]
    url = str(direct_url.get("url", ""))
    vcs = direct_url.get("vcs_info") or {}
    commit = str(vcs.get("commit_id", ""))
    if vcs.get("vcs") != "git":
        return [f"tabrion {version} is not a git install ({url}); the pin is {pin_sha[:12]}"]
    if url.removesuffix("/").removesuffix(".git") != "https://github.com/peterkarlhuber/tabrion":
        return [f"tabrion is installed from {url}, not https://github.com/peterkarlhuber/tabrion.git"]
    if commit != pin_sha:
        return [f"tabrion is installed at {commit[:12] or '-'}, requirements/tabrion.txt pins {pin_sha[:12]}"]
    expected = FINAL_TABRION_COMMITS.get(pin_sha)
    if version != expected:
        return [f"tabrion {version} is installed, the pinned commit is the final {expected}"]
    return []


def test_the_installed_tabrion_is_the_pin() -> None:
    match = PIN.fullmatch(requirement_lines(REQUIREMENTS.read_text(encoding="utf-8"))[0])
    assert match is not None
    dist = metadata.distribution("tabrion")
    text = dist.read_text("direct_url.json")
    assert installed_problems(dist.version, json.loads(text) if text else None, match["sha"]) == []


_URL = "https://github.com/peterkarlhuber/tabrion.git"


@pytest.mark.parametrize(
    ("version", "direct_url", "message"),
    [
        pytest.param("0.1.99", None, "installed from a package index", id="index"),
        pytest.param(
            "0.1.0",
            {"url": "file:///src/tabrion", "dir_info": {"editable": True}},
            "not a git install",
            id="local",
        ),
        pytest.param(
            "0.1.0",
            {"url": _URL, "vcs_info": {"vcs": "git", "commit_id": "e" * 40}},
            "installed at eeee",
            id="other-commit",
        ),
        pytest.param(
            "0.1.0",
            {"url": "https://evil.invalid/tabrion.git", "vcs_info": {"vcs": "git", "commit_id": SHA}},
            "installed from https://evil.invalid",
            id="other-url",
        ),
        pytest.param(
            "0.1.1",
            {"url": _URL, "vcs_info": {"vcs": "git", "commit_id": SHA}},
            "the final 0.1.0",
            id="other-version",
        ),
    ],
)
def test_an_installed_tabrion_other_than_the_pin_turns_red(
    version: str, direct_url: dict[str, Any] | None, message: str
) -> None:
    problems = installed_problems(version, direct_url, SHA)
    assert len(problems) == 1 and message in problems[0], problems


@pytest.mark.parametrize("url", [_URL, _URL.removesuffix(".git"), _URL + "/"])
def test_the_pin_passes_in_any_url_spelling(url: str) -> None:
    assert installed_problems("0.1.0", {"url": url, "vcs_info": {"vcs": "git", "commit_id": SHA}}, SHA) == []
