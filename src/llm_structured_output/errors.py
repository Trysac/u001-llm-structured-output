"""Typed errors raised by the library.

Every error derives from :class:`StructuredOutputError` and carries a stable, machine-readable
``code``. Messages never include reply text and are at most 200 characters long.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .models import AttemptRecord, Repair

ErrorCode = Literal[
    "input_too_large",
    "no_json",
    "invalid_json",
    "schema_validation",
    "attempts_exhausted",
]

_MAX_MESSAGE_CHARS = 200


class StructuredOutputError(Exception):
    """Base class of every error raised by this library.

    Attributes:
        code: Stable machine-readable error code.
    """

    def __init__(self, message: str, code: ErrorCode) -> None:
        super().__init__(message[:_MAX_MESSAGE_CHARS])
        self.code: ErrorCode = code


class InputTooLargeError(StructuredOutputError):
    """The text exceeds ``max_input_chars``; raised before any scanning."""

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"input has {size} characters, limit is {limit}", "input_too_large")
        self.size = size
        self.limit = limit


class NoJsonFoundError(StructuredOutputError):
    """The text contains no JSON object or array candidate."""

    def __init__(self) -> None:
        super().__init__("no JSON object found", "no_json")


class InvalidJsonError(StructuredOutputError):
    """Candidates were found but none decodes as JSON, even after repairs.

    Attributes:
        position: Character offset in the input of the first failure. It is exact when no
            repair changed lengths before it and is the end of the scanned text for
            unterminated candidates.
    """

    def __init__(self, position: int) -> None:
        super().__init__(f"invalid JSON at position {position}", "invalid_json")
        self.position = position


class SchemaValidationError(StructuredOutputError):
    """The decoded JSON does not satisfy the model.

    Attributes:
        errors: Stored ``(path, message)`` pairs (at most 1000, each part at most 200 chars).
        total: The real number of validation errors.
        repairs: Repairs that were applied before validation.
    """

    def __init__(
        self,
        errors: Sequence[tuple[str, str]],
        *,
        repairs: tuple[Repair, ...] = (),
        total: int | None = None,
    ) -> None:
        self.errors: list[tuple[str, str]] = list(errors)
        self.total = len(self.errors) if total is None else total
        self.repairs = repairs
        super().__init__(f"schema validation failed with {self.total} error(s)", "schema_validation")


class AttemptsExhaustedError(StructuredOutputError):
    """Every attempt of ``generate`` failed.

    Attributes:
        attempts: One record per model call, in order.
    """

    def __init__(self, attempts: Sequence[AttemptRecord]) -> None:
        self.attempts: tuple[AttemptRecord, ...] = tuple(attempts)
        super().__init__(f"all {len(self.attempts)} attempts failed", "attempts_exhausted")
