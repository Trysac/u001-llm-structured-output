"""Validated structured output from LLM replies into Pydantic models.

The public API is :func:`extract_json`, :func:`parse_reply`, :func:`generate`, :class:`Result`
and :class:`AttemptRecord`. The LLM port, its test doubles and the typed errors are exported too.
"""

from __future__ import annotations

import logging

from .core import generate, parse_reply
from .errors import (
    AttemptsExhaustedError,
    ErrorCode,
    InputTooLargeError,
    InvalidJsonError,
    NoJsonFoundError,
    SchemaValidationError,
    StructuredOutputError,
)
from .extract import extract_json
from .models import AttemptRecord, Repair, Result
from .ports import FakeLLM, LLMPort, Message, RecordedLLM, RecordingExhaustedError

__all__ = [
    "AttemptRecord",
    "AttemptsExhaustedError",
    "ErrorCode",
    "FakeLLM",
    "InputTooLargeError",
    "InvalidJsonError",
    "LLMPort",
    "Message",
    "NoJsonFoundError",
    "RecordedLLM",
    "RecordingExhaustedError",
    "Repair",
    "Result",
    "SchemaValidationError",
    "StructuredOutputError",
    "__version__",
    "extract_json",
    "generate",
    "parse_reply",
]

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
