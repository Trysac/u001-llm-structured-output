"""Linear-time JSON extraction and string-safe repair of LLM replies.

A single left-to-right tokenizer state machine (outside a string, inside a double-quoted
string, inside a single-quoted string, inside an escape) walks each candidate once. Repairs
only touch tokens outside strings. Inside strings the only changes are escaping raw control
characters and, when a single-quoted string is converted, escaping inner double quotes.
"""

from __future__ import annotations

import json
from typing import NamedTuple

from .errors import InputTooLargeError, InvalidJsonError, NoJsonFoundError
from .models import Repair

__all__ = ["DEFAULT_MAX_INPUT_CHARS", "extract_json"]

DEFAULT_MAX_INPUT_CHARS = 200_000

_MAX_CANDIDATES = 16
_MAX_DEPTH = 100
_FENCE = "```"
_OUTSIDE, _DOUBLE, _SINGLE, _ESCAPE = range(4)
_STRUCTURAL = frozenset({"{", "}", "[", "]", ":", ",", '"', "'"})
_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}
_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}


class _Scan(NamedTuple):
    text: str
    repairs: tuple[Repair, ...]


def _note(repairs: list[Repair], repair: Repair) -> None:
    if repair not in repairs:
        repairs.append(repair)


def _escape_control(ch: str) -> str:
    return _CONTROL_ESCAPES.get(ch) or f"\\u{ord(ch):04x}"


def _check_input(text: str, max_input_chars: int) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_input_chars, bool) or max_input_chars < 1:
        raise ValueError("max_input_chars must be a positive integer")
    if len(text) > max_input_chars:
        raise InputTooLargeError(len(text), max_input_chars)


def _find_fence(text: str) -> tuple[str, int] | None:
    """Return the content and offset of the first fenced block, or ``None``."""
    opening = text.find(_FENCE)
    if opening < 0:
        return None
    start = opening + len(_FENCE)
    if text[start : start + 4].lower() == "json":
        start += 4
    end = text.find(_FENCE, start)
    if end < 0:
        end = len(text)
    return text[start:end], start


def _next_open(text: str, pos: int) -> int:
    brace = text.find("{", pos)
    bracket = text.find("[", pos)
    if brace < 0:
        return bracket
    if bracket < 0:
        return brace
    return min(brace, bracket)


def _scan(text: str, start: int) -> _Scan | None:
    """Repair and delimit the balanced object or array that starts at ``text[start]``.

    Returns ``None`` when the candidate is unterminated, mismatched or nested too deeply.
    """
    out: list[str] = []
    repairs: list[Repair] = []
    stack: list[str] = []
    state = _OUTSIDE
    resume = _OUTSIDE
    pending_comma = -1
    value_end = False
    i = start
    size = len(text)
    while i < size:
        ch = text[i]
        if state == _OUTSIDE:
            if ch in "{[":
                if len(stack) >= _MAX_DEPTH:
                    raise InvalidJsonError(i)
                stack.append("}" if ch == "{" else "]")
                out.append(ch)
                pending_comma = -1
                value_end = False
            elif ch in "}]":
                if not stack or stack.pop() != ch:
                    return None
                if pending_comma >= 0:
                    out[pending_comma] = ""
                    _note(repairs, "trailing_comma")
                pending_comma = -1
                out.append(ch)
                value_end = True
                if not stack:
                    return _Scan("".join(out), tuple(repairs))
            elif ch == '"':
                out.append(ch)
                state = _DOUBLE
                pending_comma = -1
            elif ch == "'":
                out.append('"')
                state = _SINGLE
                pending_comma = -1
                _note(repairs, "single_quote")
            elif ch == ",":
                out.append(ch)
                pending_comma = len(out) - 1 if value_end else -1
                value_end = False
            elif ch == ":":
                out.append(ch)
                pending_comma = -1
                value_end = False
            elif ch.isspace():
                out.append(ch)
            else:
                end = i + 1
                while end < size and not text[end].isspace() and text[end] not in _STRUCTURAL:
                    end += 1
                word = text[i:end]
                replacement = _PY_LITERALS.get(word)
                if replacement is None:
                    out.append(word)
                else:
                    out.append(replacement)
                    _note(repairs, "python_literal")
                pending_comma = -1
                value_end = True
                i = end
                continue
        elif state == _DOUBLE:
            if ch == "\\":
                state = _ESCAPE
                resume = _DOUBLE
            elif ch == '"':
                out.append(ch)
                state = _OUTSIDE
                value_end = True
            elif ord(ch) < 0x20:
                out.append(_escape_control(ch))
                _note(repairs, "escape_control_char")
            else:
                out.append(ch)
        elif state == _SINGLE:
            if ch == "\\":
                state = _ESCAPE
                resume = _SINGLE
            elif ch == "'":
                out.append('"')
                state = _OUTSIDE
                value_end = True
            elif ch == '"':
                out.append('\\"')
            elif ord(ch) < 0x20:
                out.append(_escape_control(ch))
                _note(repairs, "escape_control_char")
            else:
                out.append(ch)
        else:
            if ord(ch) < 0x20:
                out.append("\\\\" + _escape_control(ch))
                _note(repairs, "escape_control_char")
            elif resume == _SINGLE and ch == "'":
                out.append(ch)
            else:
                out.append("\\" + ch)
            state = resume
        i += 1
    return None


def extract_value(text: str, max_input_chars: int) -> tuple[str, tuple[Repair, ...], object]:
    """Extract, repair and decode the first JSON candidate of ``text``.

    Returns the repaired JSON text, the repairs applied and the decoded value.
    """
    _check_input(text, max_input_chars)
    region, offset = text, 0
    prefix: tuple[Repair, ...] = ()
    fence = _find_fence(text)
    if fence is not None:
        region, offset = fence
        prefix = ("strip_fence",)
    failure: int | None = None
    pos = 0
    for _ in range(_MAX_CANDIDATES):
        start = _next_open(region, pos)
        if start < 0:
            break
        pos = start + 1
        try:
            scanned = _scan(region, start)
        except InvalidJsonError as exc:
            raise InvalidJsonError(offset + exc.position) from None
        if scanned is None:
            if failure is None:
                failure = offset + len(region)
            continue
        try:
            value = json.loads(scanned.text)
        except ValueError as exc:
            if failure is None:
                detail = exc.pos if isinstance(exc, json.JSONDecodeError) else 0
                failure = offset + start + detail
            continue
        return scanned.text, (*prefix, *scanned.repairs), value
    if failure is None:
        raise NoJsonFoundError()
    raise InvalidJsonError(failure)


def extract_json(
    text: str, *, max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
) -> tuple[str, tuple[Repair, ...]]:
    """Return the first decodable JSON object or array of ``text`` and the repairs applied.

    The first fenced block (three backticks, optional ``json`` tag) wins. Otherwise balanced
    objects or arrays are tried left to right (at most 16) and the first one that
    ``json.loads`` accepts after repairs is used. Repairs are ``strip_fence``,
    ``trailing_comma``, ``single_quote``, ``python_literal`` and ``escape_control_char``; each
    is listed once in first-applied order.

    Raises:
        InputTooLargeError: if ``text`` is longer than ``max_input_chars``.
        NoJsonFoundError: if no object or array candidate exists.
        InvalidJsonError: if candidates exist but none decodes.
        ValueError: if ``max_input_chars`` is not a positive integer.
        TypeError: if ``text`` is not a string.
    """
    json_text, repairs, _ = extract_value(text, max_input_chars)
    return json_text, repairs
