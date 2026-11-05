# Single-pass tokenizer for JSON extraction and repair

Status: accepted (generated, pending human review)

## Context

LLM replies wrap JSON in prose or code fences and contain trailing commas, single quotes and Python literals. Regex-based repairs corrupt string contents (for example turning the text True inside a string into true) and can backtrack catastrophically on adversarial input. Retry loops that resend the bad reply leak it into prompts and logs.

## Decision

Extraction and repair run in one left-to-right tokenizer state machine with four states (outside a string, double-quoted, single-quoted, escape), O(n) per candidate, with no nested-quantifier regex. Repairs touch only tokens outside strings; inside strings the only edits are escaping raw control characters and escaping inner double quotes when a single-quoted string is converted. String states start only inside a candidate, so prose apostrophes never open a string. The first fenced block wins; otherwise at most 16 balanced candidates are tried left to right and the first that json.loads accepts is used. Nesting is capped at 100 levels and input at max_input_chars (default 200,000). The re-ask loop is bounded to 1-10 attempts, appends one user message per failure listing at most 20 error lines, never resends the bad reply, and AttemptRecord stores only counts and codes, never text. The library depends only on pydantic v2 and is vendor-free.

## Consequences

Repairs are predictable, reportable (each listed once in first-applied order) and safe for string contents. Worst-case work is bounded at 16 candidate scans over a capped input. Only the first decodable candidate is used, so a leading valid array in prose can win over a later object. Position values in InvalidJsonError are approximate when repairs shifted characters. Unsupported by design: streaming, async and tool calling.
