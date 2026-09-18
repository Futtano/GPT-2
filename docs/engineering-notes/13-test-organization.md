# Organize tests by scope and package ownership

Mirroring the import package makes ownership clear, while scope directories
leave room for broader workflows:

```text
tests/
├── conftest.py
├── unit/
│   └── gpt_2/
│       ├── test_config.py
│       ├── test_dataset.py
│       ├── test_model.py
│       ├── test_pretrain.py
│       └── test_sattn.py
├── integration/
└── e2e/
```

- **Unit tests** isolate functions and classes with small local inputs.
- **Integration tests** connect real components such as configuration, data,
  training, and checkpoint storage.
- **End-to-end tests** invoke the installed CLI as a user would.
- `tests/conftest.py` contains fixtures shared across test modules.

Test directories do not need `__init__.py` unless they deliberately form
import packages. Avoid importing helpers from one test module into another;
put reusable fixtures in `conftest.py` and ordinary helper code in a clearly
named support module.

After moving tests, verify that every case is discovered exactly once:

```bash
uv run pytest --collect-only -q
uv run pytest -q -W error
```

Keep test placement aligned with behavior. Tests for `gpt_2.config` belong next
to other `gpt_2` unit tests; a CLI subprocess test belongs under `e2e/`.

## Further reading

- [pytest: Good integration practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [pytest: Sharing fixtures across files](https://docs.pytest.org/en/stable/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-or-session)

[Back to the engineering-notes index](../engineering-notes.md)
