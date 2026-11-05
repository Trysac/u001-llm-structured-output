"""The LLM port and deterministic test doubles."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol, TypedDict

Message = TypedDict("Message", {"role": Literal["system", "user", "assistant"], "content": str})  # noqa: UP013

_DEFAULT_MAX_CALLS = 1000


class LLMPort(Protocol):
    """Minimal vendor-free interface to a chat model."""

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        """Return the assistant reply for ``messages``."""


class RecordingExhaustedError(RuntimeError):
    """A ``RecordedLLM`` was called more often than it has recorded replies."""


def snapshot_messages(messages: Sequence[Message]) -> list[Message]:
    """Return an independent copy of ``messages``."""
    return [Message(role=m["role"], content=m["content"]) for m in messages]


def _check_max_calls(max_calls: int) -> None:
    if max_calls < 1:
        raise ValueError("max_calls must be at least 1")


def _record(calls: list[list[Message]], messages: Sequence[Message], max_calls: int) -> None:
    calls.append(snapshot_messages(messages))
    if len(calls) > max_calls:
        del calls[0]


class RecordedLLM:
    """Replays recorded replies in order and records every request in ``calls``.

    Only the newest ``max_calls`` requests are kept.
    """

    def __init__(self, replies: Sequence[str], *, max_calls: int = _DEFAULT_MAX_CALLS) -> None:
        _check_max_calls(max_calls)
        self._replies = tuple(replies)
        self._max_calls = max_calls
        self._next = 0
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        """Record the request and return the next reply.

        Raises:
            RecordingExhaustedError: when every recorded reply was already returned.
        """
        _record(self.calls, messages, self._max_calls)
        if self._next >= len(self._replies):
            raise RecordingExhaustedError("no recorded replies left")
        reply = self._replies[self._next]
        self._next += 1
        return reply


class FakeLLM:
    """Answers every request with ``fn(messages)`` and records it in ``calls``.

    Only the newest ``max_calls`` requests are kept.
    """

    def __init__(self, fn: Callable[[Sequence[Message]], str], *, max_calls: int = 1000) -> None:
        _check_max_calls(max_calls)
        self._fn = fn
        self._max_calls = max_calls
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        """Record the request and return ``fn(messages)``."""
        _record(self.calls, messages, self._max_calls)
        return self._fn(snapshot_messages(messages))
