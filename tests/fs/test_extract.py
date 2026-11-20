"""SafeRoot.extract_text: text files and PDFs within page, character, time and memory budgets (PS-1)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import time
from types import ModuleType

import pytest

from orkexa.fs import FsError, SafeRoot, TextBudget

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "pdf"


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_pdf_fixtures", FIXTURES / "make_pdf_fixtures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code(call: object) -> FsError:
    assert callable(call)
    with pytest.raises(FsError) as raised:
        call()
    return raised.value


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    path = tmp_path / "root"
    path.mkdir()
    return path


def test_text_files(folder: Path) -> None:
    (folder / "notes.md").write_bytes("# Title\nÄ text\n".encode())
    (folder / "bom.txt").write_bytes(b"\xef\xbb\xbfwith bom")
    (folder / "wide.txt").write_bytes("UTF-16 text".encode("utf-16"))
    (folder / "broken.txt").write_bytes(b"bad \xff byte")
    root = SafeRoot(folder)
    extracted = root.extract_text("notes.md")
    assert (extracted.kind, extracted.text, extracted.pages) == ("text", "# Title\nÄ text\n", None)
    assert root.extract_text("bom.txt").text == "with bom"
    assert root.extract_text("wide.txt").text == "UTF-16 text"
    assert root.extract_text("broken.txt").text == "bad � byte"


def test_binary_and_unknown_files_are_refused(folder: Path) -> None:
    (folder / "binary.txt").write_bytes(b"MZ\x00\x00\x01")
    (folder / "program.exe").write_bytes(b"MZ")
    (folder / "fake.pdf").write_bytes(b"not a pdf")
    root = SafeRoot(folder)
    assert _code(lambda: root.extract_text("binary.txt")).code == "fs.unsupported_type"
    assert _code(lambda: root.extract_text("program.exe")).code == "fs.unsupported_type"
    assert _code(lambda: root.extract_text("fake.pdf")).code == "fs.unreadable"


def test_the_character_and_byte_budgets(folder: Path) -> None:
    (folder / "long.txt").write_bytes(b"x" * 5000)
    root = SafeRoot(folder)
    error = _code(lambda: root.extract_text("long.txt", budget=TextBudget(max_chars=100)))
    assert (error.code, error.reason) == ("fs.too_complex", "chars")
    assert (
        _code(lambda: root.extract_text("long.txt", budget=TextBudget(max_bytes=100))).code == "fs.too_large"
    )
    with pytest.raises(ValueError):
        TextBudget(timeout=0)


def test_extraction_goes_through_the_guard(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    (tmp_path / "secret.txt").write_bytes(b"outside")
    root = SafeRoot(folder)
    assert _code(lambda: root.extract_text("../secret.txt")).code == "fs.parent_reference"
    assert _code(lambda: root.extract_text(".ssh/config.txt")).code == "fs.blocked"


@pytest.fixture
def make() -> ModuleType:
    pytest.importorskip("pypdf")
    return _generator()


def test_a_normal_pdf(folder: Path, make: ModuleType) -> None:
    shutil.copy(FIXTURES / "normal_10p.pdf", folder / "Report.PDF")
    extracted = SafeRoot(folder).extract_text("Report.PDF")
    assert extracted.kind == "pdf"
    assert list(extracted.pages or ()) == make.NORMAL_TEXTS
    assert extracted.text == "\n\n".join(make.NORMAL_TEXTS)


def test_a_pdf_bomb_ends_within_the_time_budget(folder: Path, make: ModuleType) -> None:
    """PS-1: a page that keeps pypdf busy is stopped at the time budget; the call returns in time."""
    shutil.copy(FIXTURES / "slow_content_stream.pdf", folder / "slow.pdf")
    started = time.monotonic()
    error = _code(lambda: SafeRoot(folder).extract_text("slow.pdf", budget=TextBudget(timeout=2)))
    assert time.monotonic() - started <= 2 + 5
    assert (error.code, error.reason) == ("fs.too_complex", "timeout")


def test_a_pdf_with_too_many_pages(folder: Path, make: ModuleType) -> None:
    (folder / "pages.pdf").write_bytes(make.pages_100k())
    started = time.monotonic()
    error = _code(lambda: SafeRoot(folder).extract_text("pages.pdf", budget=TextBudget(max_pages=100)))
    assert time.monotonic() - started <= 30 + 5
    assert (error.code, error.reason) == ("fs.too_complex", "pages")


def test_a_deeply_nested_pdf_ends_within_the_budget(folder: Path, make: ModuleType) -> None:
    shutil.copy(FIXTURES / "nested_objects.pdf", folder / "nested.pdf")
    started = time.monotonic()
    try:
        SafeRoot(folder).extract_text("nested.pdf", budget=TextBudget(timeout=5))
    except FsError as error:
        assert error.code in ("fs.too_complex", "fs.unreadable")
    assert time.monotonic() - started <= 5 + 5


def test_a_flate_bomb_hits_the_memory_budget(folder: Path, make: ModuleType) -> None:
    (folder / "bomb.pdf").write_bytes(make.flate_bomb())
    error = _code(lambda: SafeRoot(folder).extract_text("bomb.pdf", budget=TextBudget(max_memory_mb=256)))
    assert (error.code, error.reason) == ("fs.too_complex", "memory")
