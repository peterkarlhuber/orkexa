"""ORK3-09: build_model verifies under a temporary name; a failed build never removes the installed model.

Internal backend audit round 3 (2026-11-04). No network.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct
from typing import Any

import pytest

from orkexa.local import OllamaError
from orkexa.local.convert import SourceFile, build_model

from ..test_local_convert import StubClient, sources  # noqa: F401  (fixture)

pytest.importorskip("gguf")


class InstalledClient(StubClient):
    """A server that already has a working build of the recipe's model."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.created["tiny:1m"] = b"previous working build"


def test_a_failed_build_keeps_the_installed_model(sources: Any) -> None:  # noqa: F811
    recipe, weights, template = sources
    client = InstalledClient(vectors=[[0.9, 0.44, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(OllamaError) as error:
        build_model(client, recipe, weights=weights, template=template)  # type: ignore[arg-type]
    assert error.value.code == "reference_mismatch"
    assert client.created == {"tiny:1m": b"previous working build"}  # untouched, staging removed
    assert "tiny:1m" not in client.deleted and not client.copied


def test_an_unexpected_error_during_the_check_removes_only_the_staging_model(sources: Any) -> None:  # noqa: F811
    recipe, weights, template = sources

    class Broken(InstalledClient):
        def embed(self, model: str, inputs: list[str], **_: Any) -> Any:
            raise TypeError("unexpected answer")

    client = Broken()
    with pytest.raises(TypeError):
        build_model(client, recipe, weights=weights, template=template)  # type: ignore[arg-type]
    assert client.created == {"tiny:1m": b"previous working build"}
    assert len(client.deleted) == 1 and client.deleted[0].startswith("tiny-orkexa-build-")


def test_a_successful_build_replaces_the_installed_model(sources: Any) -> None:  # noqa: F811
    recipe, weights, template = sources
    client = InstalledClient(vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    build_model(client, recipe, weights=weights, template=template)  # type: ignore[arg-type]
    assert list(client.created) == ["tiny:1m"] and client.created["tiny:1m"].startswith(b"GGUF")


def test_oversized_inputs_and_malformed_weights_are_refused(sources: Any, tmp_path: Path) -> None:  # noqa: F811
    recipe, weights, template = sources
    huge = replace(recipe, weights=SourceFile(recipe.weights.url, recipe.weights.sha256, 9 * 1024**3))
    with pytest.raises(OllamaError) as error:
        build_model(StubClient(), huge, weights=weights, template=template)  # type: ignore[arg-type]
    assert error.value.code == "too_large"
    from orkexa.local.convert import _safetensors

    for header in (b"[1, 2]", b'{"t": {"dtype": "F32", "shape": [-1, -4], "data_offsets": [0, 16]}}', b"{"):
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 16)
        with pytest.raises(OllamaError) as error:
            list(_safetensors(bad))
        assert error.value.code == "invalid_weights"
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\1\2")
    with pytest.raises(OllamaError):
        list(_safetensors(short))
