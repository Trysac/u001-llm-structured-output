# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `extract_json`, `parse_reply` and `generate`: linear string-safe JSON extraction and repair of LLM replies into Pydantic v2 models with typed errors, a bounded re-ask loop, an LLM port and the RecordedLLM/FakeLLM test doubles.
- Initial project structure from the factory utility template.
