# llm-structured-output

[![PyPI version](https://img.shields.io/pypi/v/trysac-llm-structured-output)](https://pypi.org/project/trysac-llm-structured-output/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-llm-structured-output)](LICENSE)
[![CI](https://github.com/Trysac/u001-llm-structured-output/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u001-llm-structured-output/actions/workflows/ci.yml)

Validated structured output from LLM replies into Pydantic models, with linear JSON extraction, string-safe repairs and a bounded re-ask loop

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-llm-structured-output
# or
uv add trysac-llm-structured-output
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import llm_structured_output

# normalize() trims surrounding whitespace and lower-cases the text.
print(llm_structured_output.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    llm_structured_output.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
