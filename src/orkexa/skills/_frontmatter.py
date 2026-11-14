"""The YAML front matter of a ``SKILL.md``: a strict, small subset of YAML, parsed without a YAML library.

Agent Skills front matter is a flat mapping of strings, a one-level ``metadata`` mapping and, in some
clients, a list of tool names. This parser reads exactly that and refuses everything else with
:class:`FrontmatterError`, so no YAML feature (anchors, aliases, tags, deeper nesting, implicit typing) can
surprise a caller. Every scalar is a string: ``version: 1.0`` gives ``"1.0"``.

Supported values: plain scalars (also folded over indented continuation lines), single- and double-quoted
scalars on one line, block scalars (``|``, ``>`` with ``-``/``+`` chomping), flow sequences (``[a, "b"]``),
block sequences (``- a``) and one-level block mappings (``key: value``).
"""

from __future__ import annotations

import re

__all__ = ["FrontmatterError", "FrontmatterValue", "parse_frontmatter", "split_frontmatter"]

FrontmatterValue = str | list[str] | dict[str, str]

_KEY = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*)\s*:(?:\s+(.*))?$")
_SEQUENCE_ITEM = re.compile(r"^-(?:\s+(.*))?$")
_BLOCK_HEADER = re.compile(r"^([|>])([+-]?)\s*(?:#.*)?$")
_ESCAPES = {"\\": "\\", '"': '"', "/": "/", "n": "\n", "t": "\t", "r": "\r", "0": "\0", "b": "\b", "f": "\f"}


class FrontmatterError(ValueError):
    """Missing or unterminated front matter, or YAML this parser does not read. ``line`` is 1-based."""

    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(message if line is None else f"line {line}: {message}")
        self.line = line


def split_frontmatter(text: str) -> tuple[str, str]:
    """``(front matter, body)`` of a ``SKILL.md``, which must start with a ``---`` line (after a BOM)."""
    if text.startswith("﻿"):
        text = text[1:]
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n").rstrip() != "---":
        raise FrontmatterError("SKILL.md must start with a '---' line that opens the front matter", 1)
    for index in range(1, len(lines)):
        if lines[index].rstrip("\r\n").rstrip() == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1 :])
    raise FrontmatterError("the front matter is not closed by a '---' line")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _strip_comment(value: str) -> str:
    """A plain scalar ends before `` #``."""
    match = re.search(r"\s#", value)
    return (value[: match.start()] if match else value).rstrip()


def _double_quoted(text: str, line: int) -> tuple[str, str]:
    """The value of a double-quoted scalar at the start of ``text`` and the rest of the line."""
    out: list[str] = []
    index = 1
    while index < len(text):
        char = text[index]
        if char == '"':
            return "".join(out), text[index + 1 :]
        if char == "\\":
            index += 1
            if index >= len(text):
                break
            escape = text[index]
            if escape in _ESCAPES:
                out.append(_ESCAPES[escape])
            elif escape in "uU":
                width = 4 if escape == "u" else 8
                digits = text[index + 1 : index + 1 + width]
                if len(digits) != width or not all(c in "0123456789abcdefABCDEF" for c in digits):
                    raise FrontmatterError(f"invalid \\{escape} escape", line)
                out.append(chr(int(digits, 16)))
                index += width
            else:
                raise FrontmatterError(f"unknown escape \\{escape}", line)
        else:
            out.append(char)
        index += 1
    raise FrontmatterError("a double-quoted value must end on the same line", line)


def _single_quoted(text: str, line: int) -> tuple[str, str]:
    out: list[str] = []
    index = 1
    while index < len(text):
        char = text[index]
        if char == "'":
            if text[index + 1 : index + 2] == "'":
                out.append("'")
                index += 2
                continue
            return "".join(out), text[index + 1 :]
        out.append(char)
        index += 1
    raise FrontmatterError("a single-quoted value must end on the same line", line)


def _scalar(text: str, line: int) -> str:
    """One scalar on one line: quoted or plain. Anchors, aliases, tags and flow mappings are refused."""
    text = text.strip()
    if not text:
        return ""
    if text[0] in "&*!":
        raise FrontmatterError("anchors, aliases and tags are not supported", line)
    if text[0] in "{":
        raise FrontmatterError("flow mappings are not supported; use one 'key: value' per line", line)
    if text[0] in "@`":
        raise FrontmatterError(f"a plain value cannot start with {text[0]!r}; quote it", line)
    if text[0] in "\"'":
        value, rest = (_double_quoted if text[0] == '"' else _single_quoted)(text, line)
        rest = rest.strip()
        if rest and not rest.startswith("#"):
            raise FrontmatterError("unexpected text after a quoted value", line)
        return value
    value = _strip_comment(text)
    if re.search(r":(\s|$)", value):
        raise FrontmatterError("a plain value cannot contain ': '; quote it", line)
    return value


def _flow_sequence(text: str, line: int) -> list[str]:
    text = _strip_comment(text) if not text.rstrip().endswith("]") else text.rstrip()
    if not text.endswith("]"):
        raise FrontmatterError("a flow sequence must end on the same line", line)
    inner = text[1:-1]
    items: list[str] = []
    rest = inner.strip()
    while rest:
        if rest[0] in "\"'":
            value, rest = (_double_quoted if rest[0] == '"' else _single_quoted)(rest, line)
            rest = rest.strip()
        else:
            comma = rest.find(",")
            value, rest = (rest, "") if comma < 0 else (rest[:comma], rest[comma:])
            value = value.strip()
            if not value or value[0] in "[{&*!":
                raise FrontmatterError("nested or empty flow items are not supported", line)
        items.append(value)
        if rest.startswith(","):
            rest = rest[1:].strip()
        elif rest:
            raise FrontmatterError("expected ',' between flow items", line)
    return items


def _block_scalar(style: str, chomp: str, lines: list[str], start: int, parent: int) -> tuple[str, int]:
    """The block scalar whose content starts at ``lines[start]``, and the next line index."""
    end = start
    content_indent: int | None = None
    while end < len(lines):
        raw = lines[end]
        if raw.strip():
            indent = _indent(raw)
            if indent <= parent:
                break
            if content_indent is None:
                content_indent = indent
            elif indent < content_indent:
                break
        end += 1
    body = [raw[content_indent or 0 :] if raw.strip() else "" for raw in lines[start:end]]
    # Trailing empty lines belong to the chomping, not to the text.
    trailing = 0
    while body and not body[-1]:
        body.pop()
        trailing += 1
    if style == "|":
        text = "\n".join(body)
    else:
        parts: list[str] = []
        previous_blank = True
        for item in body:
            if not item:
                parts.append("\n")
                previous_blank = True
            elif item.startswith((" ", "\t")):
                parts.append(("" if previous_blank else "\n") + item + "\n")
                previous_blank = True
            else:
                parts.append(item if previous_blank else " " + item)
                previous_blank = False
        text = "".join(parts).rstrip("\n") if parts else ""
    if not body:
        return "", end
    if chomp == "-":
        return text, end
    if chomp == "+":
        return text + "\n" * (1 + trailing), end
    return text + "\n", end


def _value(
    text: str, lines: list[str], index: int, parent: int, *, nested: bool
) -> tuple[FrontmatterValue, int]:
    """The value after ``key:`` on line ``index`` (0-based), and the index of the next unread line."""
    number = index + 2  # 1-based, counting the opening '---'
    header = _BLOCK_HEADER.match(text)
    if header:
        return _block_scalar(header[1], header[2], lines, index + 1, parent)
    if text.startswith("["):
        return _flow_sequence(text, number), index + 1
    if text and not text.startswith("#"):
        value = _scalar(text, number)
        follow = index + 1
        if text[0] not in "\"'":
            # A plain scalar may continue on more-indented lines (folded with single spaces).
            pieces = [value]
            while follow < len(lines) and lines[follow].strip() and _indent(lines[follow]) > parent:
                pieces.append(_scalar(lines[follow], follow + 2))
                follow += 1
            value = " ".join(piece for piece in pieces if piece)
        return value, follow
    # Nothing on the line: an indented block (mapping or sequence) or an empty value.
    follow = index + 1
    while follow < len(lines) and not lines[follow].strip():
        follow += 1
    if follow >= len(lines) or _indent(lines[follow]) <= parent:
        return "", index + 1
    child = _indent(lines[follow])
    first = lines[follow].strip()
    if _SEQUENCE_ITEM.match(first):
        items: list[str] = []
        while follow < len(lines):
            raw = lines[follow]
            if not raw.strip() or raw.strip().startswith("#"):
                follow += 1
                continue
            if _indent(raw) < child:
                break
            if _indent(raw) > child:
                raise FrontmatterError("nested sequences are not supported", follow + 2)
            match = _SEQUENCE_ITEM.match(raw.strip())
            if not match:
                raise FrontmatterError("expected '- item'", follow + 2)
            item = match[1] or ""
            if item.startswith(("[", "- ")) or _KEY.match(item):
                raise FrontmatterError("sequence items must be single values", follow + 2)
            items.append(_scalar(item, follow + 2))
            follow += 1
        return items, follow
    if nested:
        raise FrontmatterError("only one level of nesting is supported (metadata: key: value)", follow + 2)
    mapping: dict[str, str] = {}
    while follow < len(lines):
        raw = lines[follow]
        if not raw.strip() or raw.strip().startswith("#"):
            follow += 1
            continue
        if _indent(raw) < child:
            break
        if _indent(raw) > child:
            raise FrontmatterError("unexpected indentation", follow + 2)
        match = _KEY.match(raw.strip())
        if not match:
            raise FrontmatterError("expected 'key: value'", follow + 2)
        if match[1] in mapping:
            raise FrontmatterError(f"duplicate key {match[1]!r}", follow + 2)
        entry, follow = _value(match[2] or "", lines, follow, child, nested=True)
        if not isinstance(entry, str):
            raise FrontmatterError(f"the value of {match[1]!r} must be a string", follow + 1)
        mapping[match[1]] = entry
    return mapping, follow


def parse_frontmatter(text: str) -> dict[str, FrontmatterValue]:
    """The keys of the front matter ``text`` (without the ``---`` lines), in file order."""
    if "\t" in "".join(line[: _indent(line) + 1] for line in text.splitlines()):
        raise FrontmatterError("tabs are not allowed for indentation")
    lines = text.splitlines()
    result: dict[str, FrontmatterValue] = {}
    index = 0
    while index < len(lines):
        raw = lines[index]
        if not raw.strip() or raw.lstrip().startswith("#"):
            index += 1
            continue
        if _indent(raw):
            raise FrontmatterError("unexpected indentation", index + 2)
        match = _KEY.match(raw)
        if not match:
            raise FrontmatterError("expected 'key: value'", index + 2)
        if match[1] in result:
            raise FrontmatterError(f"duplicate key {match[1]!r}", index + 2)
        result[match[1]], index = _value(match[2] or "", lines, index, 0, nested=False)
    return result
