# Serialize run settings and protect existing artifacts

`artifacts.py` now provides two independently testable operations:
`build_run_metadata` constructs a JSON-compatible payload, and
`write_run_metadata` serializes and writes it to a caller-supplied path.

The module depends on configuration and runtime records, not workflow
orchestration. Directory creation remains the workflow's responsibility.

## Preserve requested and effective settings

The payload has this structure:

```json
{
  "schema_version": 1,
  "config": {
    "model": {},
    "data": {},
    "training": {}
  },
  "runtime": {
    "device": "cpu",
    "seed": 123
  }
}
```

The configuration sections contain all actual dataclass fields, recursively
converted by `dataclasses.asdict`. Configuration can retain the requested
device `"auto"`, while runtime records the resolved device `"cpu"`. Convert
`torch.device` explicitly to a string; it is not directly JSON-serializable.

Keep schema version explicit so future readers can identify the payload
format. The artifact records settings, not model weights, optimizer state,
dataset identity, metrics, or the complete state needed for resume.

## Serialize before creating a file

The writer builds the payload and calls:

```python
serialized = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
```

`indent=2` makes the artifact readable, the newline follows text-file
conventions, and `allow_nan=False` rejects NaN and infinity rather than writing
nonstandard JSON numbers. Serialization happens before opening the destination,
so a serialization error cannot create an empty artifact.

Open the destination with UTF-8 and mode `"x"`. Exclusive creation rejects an
existing path with `FileExistsError`, preserving its existing contents.
It also avoids a check-then-write race that an initial `exists()` check alone
would not prevent.

This is not an atomic-write protocol. A filesystem write failure after creation
can still leave a partial file. This milestone guarantees serialization before
creation and refusal to overwrite, not rollback of every I/O failure.

The helper does not create missing parents. A missing output directory surfaces
as `FileNotFoundError` rather than silently expanding the workflow's side
effects.

## Assert the complete payload

The metadata test compares the entire dictionary:

```python
assert metadata == {
    "schema_version": 1,
    "config": asdict(run_config),
    "runtime": {"device": str(runtime.device), "seed": runtime.seed},
}
```

A loop that only checks keys present in the returned dictionary cannot detect
missing keys; an empty dictionary would pass without executing an assertion.
Whole-payload equality catches missing and unexpected fields and avoids broad
type suppressions when the return annotation is `dict[str, object]`.

Additional tests cover requested versus resolved device values, a JSON
round-trip through a real temporary file, unchanged contents after overwrite
rejection, missing-parent errors, and unchanged configuration/runtime inputs.

## Keep fixtures consistent and isolated

The runtime fixture obtains its seed from `run_config.training.seed`, so the
requested and effective seed do not contradict one another. The model fixture
is function-scoped: a mutable model should not leak parameter or mode changes
between tests through session-wide reuse.

Temporary-file fixtures return `Path` directly. An unnecessary teardown
`unlink()` could add another error if the write under test failed before
creating the file. The next test does not need manual deletion to obtain its
own isolated `tmp_path`.

## Integration status

The serialization and writer boundary is complete. Preparation now persists
`run-config.json` after creating its fresh output directory and before returning
the prepared run; see [preparation integration](32-persist-metadata-during-preparation.md).
Later training failures may leave useful configuration evidence in that run
directory; rollback is a separate policy decision.

## Further reading

- [Python dataclasses.asdict](https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict)
- [Python JSON serialization](https://docs.python.org/3/library/json.html#json.dumps)
- [Python open modes](https://docs.python.org/3/library/functions.html#open)
- [pytest fixture scope](https://docs.pytest.org/en/stable/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-or-session)

[Back to the engineering-notes index](../engineering-notes.md)
