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

[Back to the engineering-notes index](../engineering-notes.md)
