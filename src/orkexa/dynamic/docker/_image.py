"""Image identity: run the image that was checked, by its id, and pull base images only by digest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
from typing import Any

from ..errors import SandboxError
from ._container import _cli
from ._host import _dict

__all__ = ["ImageInfo", "inspect_image", "pin_base", "resolve_image_id"]

_DIGEST_REF = re.compile(r"^(?P<repository>[^@\s]+)@(?P<digest>sha256:[0-9a-f]{64})$")


@dataclass(frozen=True)
class ImageInfo:
    """What ``docker image inspect`` says about one local image."""

    id: str
    repo_digests: tuple[str, ...]
    #: The names (never the values) of the image's ``ENV``.
    env_names: tuple[str, ...]
    os: str = ""
    architecture: str = ""


def inspect_image(
    docker: str, ref: str, *, env: Mapping[str, str] | None = None, timeout: float = 30.0
) -> ImageInfo | None:
    """The local image ``ref`` (a name, tag, digest or id), or ``None`` when there is none.

    Raises :class:`SandboxError` when the CLI cannot be run or prints no JSON."""
    if not ref or ref.startswith("-"):
        raise ValueError(f"Invalid image reference {ref!r}")
    completed = _cli(docker, ["image", "inspect", ref], env, timeout)
    if completed.returncode != 0:
        return None
    try:
        data: Any = json.loads(completed.stdout.decode("utf-8", "replace") or "[]")
    except ValueError as exc:
        raise SandboxError(f"{docker} image inspect printed no JSON") from exc
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    item = data[0]
    image_id = str(item.get("Id") or "")
    if not image_id:
        return None
    variables = _dict(item.get("Config")).get("Env") or []
    return ImageInfo(
        id=image_id,
        repo_digests=tuple(str(d) for d in item.get("RepoDigests") or []),
        env_names=tuple(str(v).split("=", 1)[0] for v in variables if isinstance(v, str)),
        os=str(item.get("Os") or ""),
        architecture=str(item.get("Architecture") or ""),
    )


def resolve_image_id(docker: str, ref: str, *, env: Mapping[str, str] | None = None) -> str:
    """The id (``sha256:...``) of the local image ``ref``; containers then run exactly that image, even if
    the tag is moved later. Raises :class:`SandboxError` when the image is not there."""
    info = inspect_image(docker, ref, env=env)
    if info is None:
        raise SandboxError(f"The image {ref} is not available locally")
    return info.id


def _repository(name: str) -> str:
    """``docker.io/library/python:3.12-slim`` -> ``python``: how Docker and Podman name the same image."""
    last = name.rsplit("/", 1)
    if ":" in last[-1]:
        name = name[: len(name) - len(last[-1])] + last[-1].split(":", 1)[0]
    for prefix in ("docker.io/library/", "docker.io/", "index.docker.io/library/", "library/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def pin_base(
    docker: str, ref_with_digest: str, *, env: Mapping[str, str] | None = None, timeout: float = 600.0
) -> str:
    """Pull ``repository[:tag]@sha256:<digest>`` and check that the local image carries exactly that digest.

    Returns the image id. Raises :class:`ValueError` for a reference without a digest and
    :class:`SandboxError` when the pull fails or the image's ``RepoDigests`` do not contain the digest."""
    match = _DIGEST_REF.match(ref_with_digest)
    if match is None:
        raise ValueError(f"{ref_with_digest!r} is not pinned: use repository[:tag]@sha256:<64 hex digits>")
    pulled = _cli(docker, ["pull", ref_with_digest], env, timeout)
    if pulled.returncode != 0:
        raise SandboxError(
            f"Cannot pull {ref_with_digest}: {pulled.stderr.decode('utf-8', 'replace')[-500:]}"
        )
    info = inspect_image(docker, ref_with_digest, env=env)
    if info is None:
        raise SandboxError(f"{ref_with_digest} is not available after the pull")
    repository, digest = _repository(match["repository"]), match["digest"]
    for entry in info.repo_digests:
        name, _, found = entry.partition("@")
        if found == digest and _repository(name) == repository:
            return info.id
    raise SandboxError(f"The pulled image does not carry the digest {digest} for {repository}")
