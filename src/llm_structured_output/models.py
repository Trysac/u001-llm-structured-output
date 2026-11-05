"""Immutable value types shared by the public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

Repair = Literal[
    "strip_fence",
    "trailing_comma",
    "single_quote",
    "python_literal",
    "escape_control_char",
]

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Result(Generic[T]):
    """Outcome of a successful parse.

    Attributes:
        value: The validated model instance.
        attempts: Number of model calls used (1 for ``parse_reply``).
        repairs: Repairs applied to the reply, each listed once in first-applied order.
    """

    value: T
    attempts: int
    repairs: tuple[Repair, ...]


@dataclass(frozen=True)
class AttemptRecord:
    """Metadata about one model call made by ``generate``; it never holds reply text.

    Attributes:
        number: 1-based attempt number.
        ok: Whether the reply parsed and validated.
        error_code: Stable error code of the failure, or ``None`` on success.
        error_count: Number of validation errors (1 for non-schema failures, 0 on success).
        repairs: Repairs applied to the reply of this attempt.
        reply_chars: Length of the reply in characters.
    """

    number: int
    ok: bool
    error_code: str | None
    error_count: int
    repairs: tuple[Repair, ...]
    reply_chars: int
