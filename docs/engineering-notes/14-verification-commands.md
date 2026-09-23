# Verification commands

Use the smallest check that can answer the current question, followed by the
full suite once focused behavior passes.

## Dependencies and packaging

```bash
# Confirm that pyproject.toml and uv.lock agree
uv lock --check

# Build source and wheel distributions
uv build

# Inspect wheel metadata
unzip -p dist/*.whl '*/METADATA' | sed -n '1,30p'
```

## Tests

```bash
# Inspect collection after reorganizing tests
uv run pytest --collect-only -q

# Run one module while iterating
uv run pytest tests/unit/gpt_2/test_config.py -q -W error

# Run the complete suite and show bottlenecks
uv run pytest -q -W error --durations=10
```

`-W error` promotes unexpected warnings to failures. `--durations=10` reports
the slowest setup, call, and teardown phases.

## Static analysis

```bash
# Check only the files involved in the current change
uv run pyright src/gpt_2/dataset.py tests/unit/gpt_2/test_dataset.py

# Check the complete package and test suite
uv run pyright
```

The focused command keeps iteration fast and makes new diagnostics easier to
attribute. The complete command is the project-wide gate. Pyright is a pinned
development dependency, and its analysis scope, interpreter, Python version,
and checking mode are stored in `pyproject.toml`.

## Diff hygiene

```bash
git diff --check
```

`git diff --check` does not inspect untracked files. Check those explicitly
while they are new:

```bash
rg -n '[[:blank:]]+$' path/to/new_file.py
```

## Working rhythm

1. Run the focused test while implementing.
2. Inspect the failure and change only what its evidence supports.
3. Re-run the focused test.
4. Run the full suite once focused behavior passes.
5. Check the diff and package metadata when relevant.

## Further reading

- [pytest command-line reference](https://docs.pytest.org/en/stable/reference/reference.html#command-line-flags)
- [uv command reference](https://docs.astral.sh/uv/reference/cli/)
- [Git documentation: `git diff`](https://git-scm.com/docs/git-diff)
- [Pyright configuration](https://github.com/microsoft/pyright/blob/main/docs/configuration.md)

[Back to the engineering-notes index](../engineering-notes.md)
