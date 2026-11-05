"""Behavioural tests for the public API of llm_structured_output."""

import contextlib
import dataclasses
import json
import logging
from typing import Any

import pytest
from pydantic import BaseModel, RootModel, create_model

import llm_structured_output
from llm_structured_output import (
    AttemptRecord,
    AttemptsExhaustedError,
    FakeLLM,
    InputTooLargeError,
    InvalidJsonError,
    NoJsonFoundError,
    RecordedLLM,
    RecordingExhaustedError,
    SchemaValidationError,
    StructuredOutputError,
    extract_json,
    generate,
    parse_reply,
)


class M(BaseModel):
    a: int


class S(BaseModel):
    a: str = ""
    s: str = ""


class Flags(BaseModel):
    flag: bool
    other: bool
    n: int | None


class Nest(BaseModel):
    a: list[int]


class Inner(BaseModel):
    b: int


class Outer(BaseModel):
    items: list[Inner]


class Two(BaseModel):
    a: int
    b: int


class IntList(RootModel[list[int]]):
    pass


class IntMap(RootModel[dict[str, int]]):
    pass


class DictList(RootModel[list[dict[str, int]]]):
    pass


class MaybeBools(RootModel[list[bool | None]]):
    pass


Big = create_model("Big", **{f"f{i}": (int, ...) for i in range(25)})

SENTINEL = "SENTINEL-7f3a"
USER = [{"role": "user", "content": "Give me JSON."}]
FIRST_REASK = "Your previous reply was invalid:\n- $: no JSON object found"
SECOND_REASK = "Your previous reply was invalid:\n- a: Input should be a valid integer, unable to parse string as an integer"  # noqa: E501

NASTY = [
    "'" * 1000,
    "{" * 1000,
    "[" * 300 + "]" * 300,
    '{"a": "' * 50,
    "{'a': '",
    '{"a": "x\\',
    "}{",
    "][",
    '{"a": 1}}}',
    "```",
    "```json",
    '{"a": 1}' + "\x00",
    "{,}",
    "[,]",
    '{"a": 1,,}',
]


class Spy:
    def __init__(self) -> None:
        self.seen: list[int] = []

    def complete(self, messages: Any, *, max_tokens: int) -> str:
        self.seen.append(max_tokens)
        return '{"a": 1}'


# --- acceptance items -------------------------------------------------------------------------


def test_fenced_block_with_trailing_comma() -> None:
    result = parse_reply('Here:\n```json\n{"a": 1,}\n```', M)
    assert result.value == M(a=1)
    assert result.repairs == ("strip_fence", "trailing_comma")
    assert result.attempts == 1


def test_single_quoted_object_keeps_apostrophe_inside_double_quoted_string() -> None:
    result = parse_reply("{'a': \"don't\"}", S)
    assert result.value.a == "don't"
    assert result.repairs == ("single_quote",)


def test_python_literal_text_inside_string_is_untouched() -> None:
    result = parse_reply('{"s": "True,"}', S)
    assert result.value.s == "True,"
    assert result.repairs == ()


def test_oversized_input_is_rejected_before_scanning(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("scanner must not run")

    monkeypatch.setattr("llm_structured_output.extract._scan", boom)
    with pytest.raises(InputTooLargeError) as info:
        parse_reply("x" * 200_001, M)
    assert info.value.code == "input_too_large"
    assert info.value.size == 200_001
    assert info.value.limit == 200_000


def test_input_exactly_at_the_limit_is_accepted() -> None:
    text = '{"a": 1}'.ljust(200_000)
    assert len(text) == 200_000
    assert parse_reply(text, M).value == M(a=1)


def test_custom_input_limit() -> None:
    assert extract_json("[1]", max_input_chars=3) == ("[1]", ())
    with pytest.raises(InputTooLargeError):
        extract_json("[1]", max_input_chars=2)


def test_generate_reasks_with_the_exact_messages() -> None:
    llm = RecordedLLM(["no json here", '{"a": "x"}', '{"a": 1}'])
    result = generate(llm, USER, M)
    assert result.attempts == 3
    assert result.value == M(a=1)
    assert len(llm.calls) == 3
    assert llm.calls[0] == USER
    assert len(llm.calls[1]) == 2
    assert llm.calls[2][1] == {"role": "user", "content": FIRST_REASK}
    assert llm.calls[2][2] == {"role": "user", "content": SECOND_REASK}


def test_every_attempt_failing_raises_exhausted() -> None:
    records: list[AttemptRecord] = []
    llm = RecordedLLM(["nope", "still nope"])
    with pytest.raises(AttemptsExhaustedError) as info:
        generate(llm, USER, M, max_attempts=2, on_attempt=records.append)
    assert info.value.code == "attempts_exhausted"
    assert len(info.value.attempts) == 2
    assert tuple(records) == info.value.attempts
    assert isinstance(info.value.__cause__, NoJsonFoundError)


def test_generate_caps_error_lines_at_twenty() -> None:
    reply = json.dumps({f"f{i}": "x" for i in range(25)})
    llm = FakeLLM(lambda messages: reply)
    with pytest.raises(AttemptsExhaustedError) as info:
        generate(llm, USER, Big, max_attempts=2)
    assert len(info.value.attempts) == 2
    assert info.value.attempts[0].error_count == 25
    lines = llm.calls[1][-1]["content"].split("\n")
    assert len(lines) == 22
    assert lines[0] == "Your previous reply was invalid:"
    assert lines[1].startswith("- f0: ")
    assert lines[20].startswith("- f19: ")
    assert lines[21] == "- (5 more errors omitted)"


def test_llm_exception_propagates_and_is_not_an_attempt() -> None:
    failure = RuntimeError("provider down")
    records: list[AttemptRecord] = []

    def fn(messages: Any) -> str:
        raise failure

    with pytest.raises(RuntimeError) as info:
        generate(FakeLLM(fn), USER, M, on_attempt=records.append)
    assert info.value is failure
    assert records == []


def test_llm_exception_after_a_failed_attempt_keeps_earlier_records() -> None:
    failure = RuntimeError("provider down")
    records: list[AttemptRecord] = []
    calls: list[int] = []

    def flaky(messages: Any) -> str:
        calls.append(1)
        if len(calls) == 1:
            return "nothing useful"
        raise failure

    with pytest.raises(RuntimeError) as info:
        generate(FakeLLM(flaky), USER, M, on_attempt=records.append)
    assert info.value is failure
    assert len(records) == 1
    assert records[0].error_code == "no_json"


def test_attempt_records_never_contain_reply_text() -> None:
    replies = [
        SENTINEL + " no braces",
        json.dumps({"a": SENTINEL}),
        json.dumps({"a": 1, "note": SENTINEL}),
    ]
    llm = RecordedLLM(replies)
    records: list[AttemptRecord] = []
    result = generate(llm, USER, M, on_attempt=records.append)
    assert result.attempts == 3
    assert len(records) == 3
    assert [r.number for r in records] == [1, 2, 3]
    assert [r.ok for r in records] == [False, False, True]
    assert [r.error_code for r in records] == ["no_json", "schema_validation", None]
    assert [r.error_count for r in records] == [1, 1, 0]
    assert [r.reply_chars for r in records] == [len(r) for r in replies]
    for record in records:
        assert SENTINEL not in repr(record)
    for call in llm.calls:
        assert all(SENTINEL not in m["content"] for m in call)


def test_exhausted_error_and_messages_never_contain_reply_text() -> None:
    reply = json.dumps({"a": SENTINEL})
    llm = FakeLLM(lambda messages: reply)
    with pytest.raises(AttemptsExhaustedError) as info:
        generate(llm, USER, M, max_attempts=2)
    assert SENTINEL not in str(info.value)
    assert all(SENTINEL not in repr(a) for a in info.value.attempts)
    assert all(SENTINEL not in m["content"] for call in llm.calls for m in call)


def test_apostrophe_in_prose_opens_no_string() -> None:
    result = parse_reply('Here\'s {x} then {"a": 1}', M)
    assert result.value == M(a=1)
    assert result.repairs == ()


def test_strict_mode_rejects_coercion() -> None:
    with pytest.raises(SchemaValidationError) as info:
        parse_reply('{"a": "1"}', M, strict=True)
    assert info.value.code == "schema_validation"
    assert info.value.errors[0][0] == "a"
    assert parse_reply('{"a": "1"}', M, strict=False).value.a == 1


# --- repairs ----------------------------------------------------------------------------------


def test_extract_json_reports_repairs_in_first_applied_order() -> None:
    text, repairs = extract_json("x {'a': True,} y")
    assert text == '{"a": true}'
    assert repairs == ("single_quote", "python_literal", "trailing_comma")


def test_each_repair_is_listed_once() -> None:
    result = parse_reply("[{'a': 1,}, {'b': 2,}]", DictList)
    assert result.repairs == ("single_quote", "trailing_comma")
    assert result.value.root == [{"a": 1}, {"b": 2}]


def test_nested_trailing_commas() -> None:
    result = parse_reply('{"a": [1, 2,], }', Nest)
    assert result.value.a == [1, 2]
    assert result.repairs == ("trailing_comma",)


def test_python_literals_are_converted_outside_strings() -> None:
    result = parse_reply("{'flag': True, 'other': False, 'n': None}", Flags)
    assert result.value == Flags(flag=True, other=False, n=None)
    assert result.repairs == ("single_quote", "python_literal")


def test_python_literal_text_inside_single_quoted_string_is_untouched() -> None:
    result = parse_reply("{'a': 'None', 's': 'False'}", S)
    assert result.value == S(a="None", s="False")
    assert result.repairs == ("single_quote",)


def test_literals_next_to_punctuation() -> None:
    text, repairs = extract_json("[True,False,None]")
    assert text == "[true,false,null]"
    assert repairs == ("python_literal",)
    assert parse_reply("[True,False,None]", MaybeBools).value.root == [True, False, None]


@pytest.mark.parametrize(
    "text",
    [
        '{"a": Trueish}',
        '{"a": True1}',
        '{"a": xTrue}',
        '{"a": True.}',
        '{"a": 1True}',
        "[None1]",
    ],
)
def test_tokens_glued_to_literals_are_not_converted(text: str) -> None:
    with pytest.raises(InvalidJsonError):
        extract_json(text)


def test_raw_control_characters_inside_strings_are_escaped() -> None:
    result = parse_reply('{"a": "x\ny\tz\x01"}', S)
    assert result.value.a == "x\ny\tz\x01"
    assert result.repairs == ("escape_control_char",)


def test_raw_control_character_after_a_backslash() -> None:
    result = parse_reply('{"a": "x\\\ny"}', S)
    assert result.value.a == "x\\\ny"
    assert result.repairs == ("escape_control_char",)


def test_valid_escapes_are_preserved() -> None:
    result = parse_reply('{"a": "q\\"uote \\u00e9 \\\\"}', S)
    assert result.value.a == 'q"uote \u00e9 \\'
    assert result.repairs == ()


def test_single_quoted_string_escapes_inner_double_quotes() -> None:
    result = parse_reply("{'a': 'say \"hi\"'}", S)
    assert result.value.a == 'say "hi"'
    assert result.repairs == ("single_quote",)


def test_single_quoted_string_unescapes_apostrophes_and_keeps_other_escapes() -> None:
    result = parse_reply("{'a': 'it\\'s', 's': 'x\\ny'}", S)
    assert result.value == S(a="it's", s="x\ny")
    assert result.repairs == ("single_quote",)


def test_raw_newline_in_single_quoted_string() -> None:
    result = parse_reply("{'a': 'x\ny'}", S)
    assert result.value.a == "x\ny"
    assert result.repairs == ("single_quote", "escape_control_char")


def test_prose_apostrophes_around_a_single_quoted_object() -> None:
    result = parse_reply("It's fine, don't worry: {'a': 'x'} isn't it", S)
    assert result.value.a == "x"
    assert result.repairs == ("single_quote",)


# --- extraction edge cases --------------------------------------------------------------------


def test_braces_inside_strings_do_not_confuse_balancing() -> None:
    result = parse_reply('{"a": "}{][", "s": "{"}', S)
    assert result.value == S(a="}{][", s="{")
    assert result.repairs == ()


def test_first_of_two_objects_wins() -> None:
    assert parse_reply('{"a": 1} {"a": 2}', M).value == M(a=1)


def test_root_model_arrays() -> None:
    result = parse_reply("noise [1, 2, 3,] noise", IntList)
    assert result.value.root == [1, 2, 3]
    assert result.repairs == ("trailing_comma",)


@pytest.mark.parametrize("text", ["", "   \n", "no braces at all"])
def test_replies_without_json_raise_no_json(text: str) -> None:
    with pytest.raises(NoJsonFoundError) as info:
        parse_reply(text, M)
    assert info.value.code == "no_json"
    assert str(info.value) == "no JSON object found"


def test_unicode_is_preserved() -> None:
    text = '{"a": "h\u00e9llo \u2713 \u65e5\u672c \U0001f600"}'
    assert parse_reply(text, S).value.a == "h\u00e9llo \u2713 \u65e5\u672c \U0001f600"
    assert parse_reply("{'a': '\u00e9'}", S).value.a == "\u00e9"


def test_invalid_json_reports_position() -> None:
    with pytest.raises(InvalidJsonError) as info:
        parse_reply('ab {"a": }', M)
    assert info.value.code == "invalid_json"
    assert info.value.position == 9
    assert str(info.value) == "invalid JSON at position 9"


def test_truncated_and_mismatched_candidates_are_invalid_json() -> None:
    with pytest.raises(InvalidJsonError) as truncated:
        parse_reply('{"a": 1', M)
    assert truncated.value.position == 7
    with pytest.raises(InvalidJsonError) as mismatched:
        parse_reply('{"a": 1]', M)
    assert mismatched.value.position == 8


def test_nesting_depth_is_limited() -> None:
    deep = "[" * 100 + "]" * 100
    assert extract_json(deep) == (deep, ())
    with pytest.raises(InvalidJsonError):
        extract_json("[" * 101 + "]" * 101)


def test_at_most_sixteen_candidates_are_tried() -> None:
    assert parse_reply("{x} " * 15 + '{"a": 1}', M).value == M(a=1)
    with pytest.raises(InvalidJsonError):
        parse_reply("{x} " * 16 + '{"a": 1}', M)


def test_fence_variants() -> None:
    assert parse_reply('```\n{"a": 1}\n```', M).repairs == ("strip_fence",)
    assert parse_reply('```JSON\n{"a": 1}', M).repairs == ("strip_fence",)


def test_first_fenced_block_wins_over_an_earlier_object() -> None:
    result = parse_reply('{"a": 2}\n```json\n{"a": 1}\n```', M)
    assert result.value == M(a=1)
    assert result.repairs == ("strip_fence",)


def test_empty_fence_means_no_json() -> None:
    with pytest.raises(NoJsonFoundError):
        parse_reply("``` ```", M)
    with pytest.raises(NoJsonFoundError):
        parse_reply('{"a": 1}\n``` ```', M)


def test_large_valid_input_is_parsed() -> None:
    text = '{"a": "' + "x" * 190_000 + '"}'
    assert len(parse_reply(text, S).value.a) == 190_000


@pytest.mark.parametrize("text", NASTY)
def test_hostile_inputs_only_raise_typed_errors(text: str) -> None:
    with contextlib.suppress(StructuredOutputError):
        parse_reply(text, M)


@pytest.mark.parametrize("text", ["{,}", "[,]", '{"a": 1,,}'])
def test_stray_commas_are_not_repaired(text: str) -> None:
    with pytest.raises(InvalidJsonError):
        extract_json(text)


def test_argument_validation() -> None:
    with pytest.raises(TypeError):
        extract_json(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_input_chars"):
        extract_json("[]", max_input_chars=0)
    with pytest.raises(ValueError, match="max_input_chars"):
        extract_json("[]", max_input_chars=True)
    with pytest.raises(TypeError):
        parse_reply("{}", dict)  # type: ignore[type-var]
    with pytest.raises(TypeError):
        parse_reply("{}", "M")  # type: ignore[type-var]


# --- schema errors ----------------------------------------------------------------------------


def test_schema_error_paths_are_joined_with_dots() -> None:
    with pytest.raises(SchemaValidationError) as info:
        parse_reply('{"items": [{"b": "x"}]}', Outer)
    path, message = info.value.errors[0]
    assert path == "items.0.b"
    assert message.startswith("Input should be a valid integer")
    assert info.value.total == 1


def test_schema_error_root_path_is_dollar() -> None:
    with pytest.raises(SchemaValidationError) as info:
        parse_reply('{"a": 1}', IntList)
    assert info.value.errors[0][0] == "$"


def test_schema_error_carries_repairs() -> None:
    with pytest.raises(SchemaValidationError) as info:
        parse_reply("{'a': 'x'}", M)
    assert info.value.repairs == ("single_quote",)


def test_schema_error_storage_and_message_are_bounded() -> None:
    with pytest.raises(SchemaValidationError) as many:
        parse_reply(json.dumps(["x"] * 1500), IntList)
    assert len(many.value.errors) == 1000
    assert many.value.total == 1500
    with pytest.raises(SchemaValidationError) as long_key:
        parse_reply(json.dumps({"k" * 500: "x"}), IntMap)
    path = long_key.value.errors[0][0]
    assert path.startswith("kkk")
    assert len(path) <= 200
    assert len(str(long_key.value)) <= 200


# --- generate ---------------------------------------------------------------------------------


def test_generate_success_on_first_attempt_reports_repairs() -> None:
    records: list[AttemptRecord] = []
    reply = "```json\n{'a': 1}\n```"
    result = generate(RecordedLLM([reply]), USER, M, on_attempt=records.append)
    assert result.attempts == 1
    assert result.repairs == ("strip_fence", "single_quote")
    assert records == [AttemptRecord(1, True, None, 0, ("strip_fence", "single_quote"), len(reply))]


def test_generate_counts_schema_errors_in_records() -> None:
    records: list[AttemptRecord] = []
    llm = RecordedLLM(['{"a": "x", "b": "y"}', '{"a": 1, "b": 2}'])
    result = generate(llm, USER, Two, on_attempt=records.append)
    assert result.attempts == 2
    assert records[0].error_count == 2
    assert records[0].error_code == "schema_validation"


def test_generate_reports_invalid_json_position() -> None:
    with pytest.raises(InvalidJsonError) as direct:
        parse_reply('{"a": }', M)
    assert direct.value.position == 6
    llm = RecordedLLM(['{"a": }', '{"a": 1}'])
    assert generate(llm, USER, M).attempts == 2
    expected = "Your previous reply was invalid:\n- $: invalid JSON at position 6"
    assert llm.calls[1][-1]["content"] == expected


def test_generate_treats_an_oversized_reply_as_a_failed_attempt() -> None:
    llm = RecordedLLM(["x" * 200_001, '{"a": 1}'])
    records: list[AttemptRecord] = []
    result = generate(llm, USER, M, on_attempt=records.append)
    assert result.attempts == 2
    assert records[0].error_code == "input_too_large"
    assert records[0].reply_chars == 200_001
    expected = "Your previous reply was invalid:\n- $: reply exceeds 200000 characters"
    assert llm.calls[1][-1]["content"] == expected


def test_generate_omitted_errors_count_is_the_real_total() -> None:
    reply = json.dumps(["x"] * 1500)
    llm = FakeLLM(lambda messages: reply)
    with pytest.raises(AttemptsExhaustedError):
        generate(llm, USER, IntList, max_attempts=2)
    lines = llm.calls[1][-1]["content"].split("\n")
    assert len(lines) == 22
    assert lines[-1] == "- (1480 more errors omitted)"


def test_generate_does_not_mutate_the_input_messages() -> None:
    messages = [{"role": "user", "content": "Give me JSON."}]
    llm = RecordedLLM(["nope", '{"a": 1}'])
    generate(llm, messages, M)
    assert messages == USER
    assert llm.calls[0] is not messages


def test_generate_passes_max_tokens_to_the_port() -> None:
    spy = Spy()
    assert generate(spy, USER, M, max_tokens=77).attempts == 1
    assert spy.seen == [77]


@pytest.mark.parametrize("attempts", [0, 11, -1, True])
def test_generate_rejects_bad_max_attempts(attempts: int) -> None:
    llm = RecordedLLM(['{"a": 1}'])
    with pytest.raises(ValueError, match="max_attempts"):
        generate(llm, USER, M, max_attempts=attempts)
    assert llm.calls == []


def test_generate_accepts_the_attempt_bounds() -> None:
    assert generate(RecordedLLM(['{"a": 1}']), USER, M, max_attempts=1).attempts == 1
    llm = FakeLLM(lambda messages: "nothing")
    with pytest.raises(AttemptsExhaustedError) as info:
        generate(llm, USER, M, max_attempts=10)
    assert len(info.value.attempts) == 10
    assert len(llm.calls) == 10


def test_generate_rejects_bad_max_tokens_and_model() -> None:
    llm = RecordedLLM(['{"a": 1}'])
    with pytest.raises(ValueError, match="max_tokens"):
        generate(llm, USER, M, max_tokens=0)
    with pytest.raises(TypeError):
        generate(llm, USER, dict)  # type: ignore[type-var]
    assert llm.calls == []


def test_generate_rejects_non_string_replies() -> None:
    llm = FakeLLM(lambda messages: 5)  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError):
        generate(llm, USER, M)


# --- doubles, errors, results, logging --------------------------------------------------------


def test_recorded_llm_replays_in_order_and_records_calls() -> None:
    llm = RecordedLLM(["one", "two"])
    assert llm.complete(USER, max_tokens=5) == "one"
    assert llm.complete(USER, max_tokens=5) == "two"
    assert llm.calls == [USER, USER]
    with pytest.raises(RecordingExhaustedError):
        llm.complete(USER, max_tokens=5)
    assert len(llm.calls) == 3


def test_fake_llm_returns_fn_result_and_records_calls() -> None:
    llm = FakeLLM(lambda messages: str(len(messages)))
    assert llm.complete(USER, max_tokens=5) == "1"
    assert llm.calls == [USER]


def test_doubles_record_snapshots() -> None:
    llm = FakeLLM(lambda messages: "ok")
    messages = [{"role": "user", "content": "x"}]
    llm.complete(messages, max_tokens=1)
    messages[0]["content"] = "changed"
    assert llm.calls[0][0]["content"] == "x"


def test_doubles_evict_the_oldest_calls() -> None:
    for llm in (RecordedLLM(["a"] * 5, max_calls=2), FakeLLM(lambda m: "a", max_calls=2)):
        for i in range(5):
            llm.complete([{"role": "user", "content": str(i)}], max_tokens=1)
        assert [call[0]["content"] for call in llm.calls] == ["3", "4"]


def test_doubles_reject_a_non_positive_max_calls() -> None:
    with pytest.raises(ValueError, match="max_calls"):
        RecordedLLM([], max_calls=0)
    with pytest.raises(ValueError, match="max_calls"):
        FakeLLM(lambda messages: "", max_calls=0)


def test_error_codes_and_hierarchy() -> None:
    errors = [
        (InputTooLargeError(5, 3), "input_too_large"),
        (NoJsonFoundError(), "no_json"),
        (InvalidJsonError(4), "invalid_json"),
        (SchemaValidationError([("a", "bad")]), "schema_validation"),
        (AttemptsExhaustedError([]), "attempts_exhausted"),
    ]
    for error, code in errors:
        assert isinstance(error, StructuredOutputError)
        assert error.code == code
    assert errors[3][0].total == 1
    assert errors[0][0].args == ("input has 5 characters, limit is 3",)


def test_base_error_truncates_long_messages() -> None:
    error = StructuredOutputError("x" * 500, "no_json")
    assert len(str(error)) == 200


def test_results_and_records_are_immutable() -> None:
    result = parse_reply('{"a": 1}', M)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.attempts = 2  # type: ignore[misc]
    record = AttemptRecord(1, True, None, 0, (), 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.ok = False  # type: ignore[misc]


def test_logging_is_silent_by_default_and_never_includes_replies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("llm_structured_output")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    with caplog.at_level(logging.DEBUG, logger="llm_structured_output"):
        generate(RecordedLLM([SENTINEL + " nothing", '{"a": 1}']), USER, M)
    assert caplog.records
    assert SENTINEL not in caplog.text


def test_package_exports_version() -> None:
    assert llm_structured_output.__version__ == "0.1.0"
