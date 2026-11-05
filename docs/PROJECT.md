# Project record

## Objective

Validated structured output from LLM replies into Pydantic models, with linear JSON extraction, string-safe repairs and a bounded re-ask loop

## Scope

- In scope: the public API documented in [README.md](../README.md), typed and tested.
- Out of scope: anything that needs runtime dependencies or I/O without an accepted ADR.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | 448402 / 102998 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.9279 |

## Factory run u001 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | claude-sonnet-5-5 | yes | 2590 | 85978 | 4246 | 0 | 0.4410 |
| fix-1 | claude-sonnet-5-5 | no | 4 | 306 | 15237 | 14903 | 0.0427 |
| fix-2 | claude-sonnet-5-5 | no | 10 | 1682 | 10629 | 80250 | 0.0514 |
| fix-3 | claude-sonnet-5-5 | no | 8 | 1388 | 7598 | 56214 | 0.0385 |
| audit | claude-sonnet-5-5 | no | 12 | 4996 | 52138 | 133881 | 0.1937 |
| fix-4 | claude-sonnet-5-5 | no | 6 | 2672 | 24528 | 56515 | 0.0937 |
| docs | claude-haiku-5-5 | no | 104009 | 5976 | 0 | 0 | 0.0669 |

Total: $0.9279. Prices: Anthropic list prices.
