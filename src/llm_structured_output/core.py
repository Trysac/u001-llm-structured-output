"""Parsing of replies into models and the bounded re-ask loop."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence

from pydantic import BaseModel, ValidationError

from .errors import (
    AttemptsExhaustedError,
    InputTooLargeError,
    InvalidJsonError,
    SchemaValidationError,
    StructuredOutputError,
)
from .extract import DEFAULT_MAX_INPUT_CHARS, extract_value
from .models import AttemptRecord, Repair, Result, T
from .ports import LLMPort, Message, snapshot_messages

__all__ = ["generate", "parse_reply"]

logger = logging.getLogger(__name__)

_MAX_ERRORS = 1000
_MAX_ERROR_LINES = 20
_MAX_FIELD_CHARS = 200
_REASK_HEADER = "Your previous reply was invalid:"


def _clip(value: str) -> str:
    return value[:_MAX_FIELD_CHARS]


def _one_line(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")


def _path(loc: Iterable[int | str]) -> str:
    return ".".join(str(part) for part in loc) or "$"


def _schema_error(exc: ValidationError, repairs: tuple[Repair, ...]) -> SchemaValidationError:
    details = exc.errors()
    errors = [(_clip(_path(e["loc"])), _clip(e["msg"])) for e in details[:_MAX_ERRORS]]
    return SchemaValidationError(errors, repairs=repairs, total=len(details))


def _require_model(model: object) -> None:
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError("model must be a pydantic BaseModel subclass")


def parse_reply(
    text: str,
    model: type[T],
    *,
    strict: bool = False,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
) -> Result[T]:
    """Turn a raw LLM reply into a validated ``model`` instance.

    The reply goes through :func:`extract_json` and then ``model.model_validate``.

    Raises:
        InputTooLargeError: if ``text`` is longer than ``max_input_chars``.
        NoJsonFoundError: if the reply holds no JSON object or array.
        InvalidJsonError: if no candidate decodes as JSON.
        SchemaValidationError: if the JSON does not satisfy ``model``.
        TypeError: if ``model`` is not a pydantic model class or ``text`` is not a string.
        ValueError: if ``max_input_chars`` is not a positive integer.
    """
    _require_model(model)
    _, repairs, payload = extract_value(text, max_input_chars)
    try:
        value = model.model_validate(payload, strict=strict)
    except ValidationError as exc:
        raise _schema_error(exc, repairs) from exc
    return Result(value=value, attempts=1, repairs=repairs)


def _describe(exc: StructuredOutputError) -> tuple[list[str], int]:
    """Return the error lines (at most 20) and the real error count for ``exc``."""
    if isinstance(exc, SchemaValidationError):
        shown = exc.errors[:_MAX_ERROR_LINES]
        return [_one_line(f"{path}: {message}") for path, message in shown], exc.total
    if isinstance(exc, InvalidJsonError):
        return [f"$: invalid JSON at position {exc.position}"], 1
    if isinstance(exc, InputTooLargeError):
        return [f"$: reply exceeds {exc.limit} characters"], 1
    return ["$: no JSON object found"], 1


def _reask_message(exc: StructuredOutputError) -> Message:
    lines, total = _describe(exc)
    content = _REASK_HEADER + "".join(f"\n- {line}" for line in lines)
    if total > len(lines):
        content += f"\n- ({total - len(lines)} more errors omitted)"
    return Message(role="user", content=content)


def _failure_record(number: int, exc: StructuredOutputError, reply_chars: int) -> AttemptRecord:
    if isinstance(exc, SchemaValidationError):
        return AttemptRecord(number, False, exc.code, exc.total, exc.repairs, reply_chars)
    return AttemptRecord(number, False, exc.code, 1, (), reply_chars)


def generate(
    llm: LLMPort,
    messages: Sequence[Message],
    model: type[T],
    *,
    max_attempts: int = 3,
    max_tokens: int = 1024,
    strict: bool = False,
    on_attempt: Callable[[AttemptRecord], None] | None = None,
) -> Result[T]:
    """Ask ``llm`` for a reply and re-ask with the validation errors until it parses.

    After each failed attempt one user message is appended that lists the errors (at most 20
    lines); the bad reply itself is never sent back. Exceptions raised by ``llm.complete``
    propagate unchanged and do not count as attempts.

    Raises:
        AttemptsExhaustedError: if every attempt failed.
        ValueError: if ``max_attempts`` is outside 1 to 10 or ``max_tokens`` is below 1.
        TypeError: if ``model`` is not a pydantic model class or the reply is not a string.
    """
    if isinstance(max_attempts, bool) or not 1 <= max_attempts <= 10:
        raise ValueError("max_attempts must be between 1 and 10")
    if isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    _require_model(model)
    conversation = snapshot_messages(messages)
    records: list[AttemptRecord] = []
    last_error: StructuredOutputError | None = None
    for number in range(1, max_attempts + 1):
        reply = llm.complete(list(conversation), max_tokens=max_tokens)
        if not isinstance(reply, str):
            raise TypeError("LLM reply must be a string")
        try:
            result = parse_reply(reply, model, strict=strict)
        except StructuredOutputError as exc:
            last_error = exc
            record = _failure_record(number, exc, len(reply))
            records.append(record)
            logger.debug("attempt %d failed with %s", number, exc.code)
            if on_attempt is not None:
                on_attempt(record)
            if number < max_attempts:
                conversation.append(_reask_message(exc))
            continue
        record = AttemptRecord(number, True, None, 0, result.repairs, len(reply))
        logger.debug("attempt %d succeeded", number)
        if on_attempt is not None:
            on_attempt(record)
        return Result(value=result.value, attempts=number, repairs=result.repairs)
    raise AttemptsExhaustedError(records) from last_error
