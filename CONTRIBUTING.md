# Contributing

Contributions are welcome, in particular:

* **adapters** for further regulators or report repositories (see `docs/sources.md`);
* **annotated sentences** that can be shared under an open licence, to improve
  the rule and supervised backends;
* **lexicon** improvements (new terms, disambiguation rules) with examples.

## Development setup

```bash
pip install -e ".[ml,pdf,dev]"
ruff check src tests
python -m pytest --cov=aidisclose
```

## Guidelines

* Open an issue before large changes.
* Add tests for new behaviour; tests must run offline (use the fixtures and the
  `FakeSession` helper in `tests/helpers.py` instead of network calls).
* Changes to the lexicon, the codebook, the rules or the prompt can change
  measurements; identify the affected component and record its version.
* Do not commit downloaded filings unless their licence allows redistribution.
