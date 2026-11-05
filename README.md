# llm-structured-output

[![PyPI version](https://img.shields.io/pypi/v/trysac-llm-structured-output)](https://pypi.org/project/trysac-llm-structured-output/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-llm-structured-output)](LICENSE)
[![CI](https://github.com/Trysac/u001-llm-structured-output/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u001-llm-structured-output/actions/workflows/ci.yml)

Validated structured output from LLM replies into Pydantic models, with linear JSON extraction, string-safe repairs and a bounded re-ask loop

## Problem

LLM replies that should be JSON arrive wrapped in prose or code fences, with trailing commas,
Python literals or fields that fail validation. Application code grows ad hoc regex repairs that
corrupt string contents and unbounded retry loops that leak replies into logs. This library turns a
raw reply into a validated Pydantic model through one linear, string-aware pass, or fails with a
typed error after a bounded re-ask loop.

## Install

```sh
pip install trysac-llm-structured-output
# or
uv add trysac-llm-structured-output
```

Requires Python 3.10 or newer and pydantic>=2.7,<3.

## Usage

```python
from pydantic import BaseModel

from llm_structured_output import RecordedLLM, generate, parse_reply


class M(BaseModel):
    a: int


result = parse_reply('```json\n{"a": 1,}\n```', M)
print(result.value, result.repairs)  # a=1 ('strip_fence', 'trailing_comma')

# Re-ask loop: invalid replies are answered with the list of errors.
llm = RecordedLLM(["no json here", '{"a": 1}'])
result = generate(llm, [{"role": "user", "content": "Give me JSON."}], M)
print(result.attempts)  # 2
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
