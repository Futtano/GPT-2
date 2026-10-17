# Verify the real workflow and collect all test scopes

`tests/integration/gpt_2/test_workflow.py` exercises `run_pretraining` without
mocking its dependencies. It complements unit tests of individual operations
and mocked assembly tests that verify forwarding, order, and failure paths.

## Run the complete in-memory path on tiny CPU inputs

The test writes a local UTF-8 input file and a complete TOML configuration
under `tmp_path`, then passes their paths and a fresh output path to the real
workflow. It exercises:

1. File and configuration validation.
2. Runtime device resolution and seeding.
3. Real GPT-2 tokenization and disjoint token splitting.
4. Windowing and all loader construction.
5. Model and optimizer construction.
6. Real optimization and periodic/final evaluation.

The model uses context length eight, embedding dimension eight, two attention
heads, one layer, no dropout, and the GPT-2 vocabulary size of 50,257. The
training configuration selects CPU, batch size two, one epoch, and zero loader
workers. Repeated short text provides enough tokens for every split to form
at least one complete input-target window.

Tiny architecture dimensions reduce computation while preserving real tensor
operations and tokenizer compatibility. There is no need to download pretrained
model weights. GPT-2 encoding assets are still required; CI prepares them
before pytest, and a fresh local run may fetch them on first use.

## Assert behavior without overfitting a smoke test

The test verifies that an output directory was created, one epoch completed,
and both update and token counts are positive. It checks that evaluation
records exist and that the final record agrees with the final epoch, step,
and token counters. Every recorded training and validation loss must be finite.

It does not require loss to decrease during a tiny run. That would test a
training-quality assumption rather than successful component integration.
Focused unit tests already establish exact schedules, token counts, and
parameter updates for known fixtures.

The checkpoint is limited to the in-memory Python workflow. It does not verify
an installed CLI, a built wheel, saved metrics or checkpoints, model selection,
resume behavior, or final test-set evaluation. The output directory exists but
artifact persistence is a subsequent milestone.

## Use temporary paths for test-owned files

`tmp_path` provides an isolated directory for each test. Inputs, configuration,
and output paths are supplied explicitly rather than relying on a checkout's
root directories.

Fixtures that only write and return a temporary file can return `Path`
directly. Yield fixtures are useful when teardown is actually needed; explicit
unlinking is not required solely because these files are under `tmp_path`.
Either style can work. pytest may retain recent temporary directories for
debugging; tests should not rely on immediate deletion after execution.

## Configure import mode when filenames repeat across test scopes

The integration test passed alone, but full-suite collection initially failed
because both unit and integration scopes contain `test_workflow.py`. Under
pytest's default import mode, these non-package test files can be imported
under the same top-level module name.

The project now stores this policy in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = ["--import-mode=importlib"]
```

Importlib mode lets pytest collect same-named files in different directories
without adding test directories to `sys.path`. Keep shared fixtures in the
appropriate `conftest.py` rather than importing one test module from another.
The application remains available through the installed project environment.

Removing bytecode caches is not a durable fix for this naming conflict.
Changing the import policy addresses the way the files are loaded.

## Verify the focused test and the complete collection

```bash
uv run --locked pytest tests/integration/gpt_2/test_workflow.py -q -W error
uv run --locked pytest -q -W error
```

The complete-suite command is necessary: running only the new file cannot
expose its import-name collision with another test scope. CI already runs the
whole suite, so integration tests are included without another workflow job.

## Further reading

- [pytest integration practices and import modes](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [pytest temporary directories](https://docs.pytest.org/en/stable/how-to/tmp_path.html)
- [Test organization](13-test-organization.md)
- [Mocked workflow assembly](29-assembled-workflow-and-cross-function-mocks.md)

[Back to the engineering-notes index](../engineering-notes.md)
