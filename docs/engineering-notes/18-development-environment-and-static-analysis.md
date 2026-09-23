# Make the development environment reproducible

Editor feedback is part of the development toolchain. If VS Code, a terminal,
and continuous integration analyze the project with different interpreters or
checker settings, developers can see different errors for the same checkout.
A production-quality project records these choices in version-controlled
configuration.

## Pin the type checker with the development dependencies

Pylance uses Pyright as its type-checking engine. Adding Pyright to the `uv`
development dependency group makes the checker version reproducible and gives
the project a command-line check that does not depend on one editor:

```toml
[dependency-groups]
dev = [
    "pyright>=1.1.414",
    "pytest>=9.1.1",
    "pytest-mock>=3.15.1",
]
```

After cloning the repository, create or synchronize the environment with
`uv sync`. In VS Code, select `.venv/bin/python` as the Python interpreter and
reload the window after changing environments.

## Store the analysis scope and Python version

The project keeps its Pyright settings in `pyproject.toml`:

```toml
[tool.pyright]
include = ["src", "tests"]
exclude = ["notebooks"]
venvPath = "."
venv = ".venv"
pythonVersion = "3.13"
typeCheckingMode = "standard"
```

The package and tests are production code and belong in the analysis scope.
The notebooks are exploratory copies and are excluded so duplicated teaching
code does not obscure diagnostics in the installed package. Exclusion should
follow repository ownership boundaries; it should not be used to hide errors
in production modules.

`pythonVersion` makes syntax and standard-library checks consistent.
`venvPath` and `venv` tell command-line Pyright where installed dependencies
live. The editor should still select the same interpreter so execution,
completion, and analysis agree.

## Distinguish environment failures from code failures

Many unrelated `reportMissingImports` errors appearing at once usually mean
the checker selected the wrong interpreter or could not find its site-packages.
Confirm that the selected interpreter can import the dependency before changing
application code:

```bash
.venv/bin/python -c "import torch, tiktoken, pytest"
```

Once imports resolve, diagnostics about optional values, invalid attribute
access, or incompatible arguments are normally code-level findings. Fix the
types or control flow instead of disabling the diagnostic globally.

For example, PyTorch exposes `DataLoader.dataset` through the broad `Dataset`
base type, which does not statically promise `__len__`. The application knows
that its factory installs `GPTTokenDataset`, so a runtime assertion both checks
the invariant and narrows the type for Pyright:

```python
dataset = loader.dataset
assert isinstance(dataset, GPTTokenDataset)

dataset_size = len(dataset)
```

This is preferable to `# type: ignore` because the assertion fails if the
factory contract changes unexpectedly.

## Use focused checks while iterating

Run Pyright on the files being changed before scanning the entire project:

```bash
uv run pyright src/gpt_2/dataset.py tests/unit/gpt_2/test_dataset.py
uv run pyright
```

The focused loader check currently reports zero errors. The repository-wide
check still identifies nine earlier diagnostics in `classification_ft.py`,
`gpt_download.py`, `instruction_ft.py`, and `sattn.py`. These remain visible as
a cleanup target rather than being silenced through configuration.

## Further reading

- [Pyright configuration](https://github.com/microsoft/pyright/blob/main/docs/configuration.md)
- [VS Code Python environments](https://code.visualstudio.com/docs/python/environments)
- [Python typing guide: type narrowing](https://typing.python.org/en/latest/guides/type_narrowing.html)

[Back to the engineering-notes index](../engineering-notes.md)
