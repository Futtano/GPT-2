# Persist run metadata during preparation

`prepare_pretrain_run` now writes `run-config.json` after creating the fresh
output directory and constructing `ResolvedRuntime`, before returning the
prepared run. Successful runs retain validated settings alongside the output
location; later data-loading or training failures can leave the same useful
configuration evidence.

## Preserve the early failure boundary

Missing configuration or input files, invalid configuration, existing output
paths, and device-resolution failures still stop before artifact writing.
Each corresponding workflow unit test patches
`gpt_2.workflow.write_run_metadata` and asserts it was not called.

The successful preparation test verifies the exact destination, configuration,
and resolved runtime passed to the writer. It uses a callable side effect to
assert that the destination's parent directory exists when writing begins.

An assertion after preparation returns only proves the directory exists at
the end. It cannot detect an implementation that calls a mocked writer before
creating the directory. The callback checks the required ordering at the
moment the dependent operation runs.

## Let write failures propagate

The writer-failure test supplies `OSError("metadata write failed")` as the
mock's side effect. Preparation propagates that failure rather than returning
a seemingly successful prepared run. The test also verifies the writer's
arguments and that the newly created output directory remains.

Preparation is not transactional. A write failure may leave a directory or
partial file, and retrying the same output path remains prohibited by the
fresh-directory policy. This milestone does not automatically delete artifacts
after failure.

## Verify persisted JSON against independent input

The real CPU integration test now reads `run-config.json` and compares the
complete payload with:

```python
{
    "schema_version": 1,
    "config": tomllib.loads(TOML_CONFIG),
    "runtime": {"device": "cpu", "seed": 123},
}
```

The expected configuration comes from the test's input TOML, not from
`build_run_metadata`. That avoids using the implementation under test to
generate its own expected answer. Whole-payload equality covers all sections
and fields, including unexpected or missing keys.

The integration test exercises the actual writer with no artifact mock.
Unit tests establish call order and failure behavior; integration establishes
that real preparation creates a valid persisted artifact and training still
completes.

## Further reading

- [Versioned metadata and exclusive writes](31-run-metadata-and-exclusive-json-writes.md)
- [Inspect temporary state with mock callbacks](26-mode-preserving-evaluation.md)
- [Real workflow integration](30-real-workflow-integration-and-test-imports.md)
- [Python TOML parsing](https://docs.python.org/3/library/tomllib.html)

[Back to the engineering-notes index](../engineering-notes.md)
