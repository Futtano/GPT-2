# Compose the workflow and test returned objects

`create_pretrain_loaders` in `workflow.py` connects existing boundaries. It
accepts `PreparedPretrainRun`, loads token IDs from its input path, forwards the
model, training, and data configuration to `create_data_loaders`, and returns
the resulting `DataLoaderBundle`.

## Give orchestration a small responsibility

The dataset module owns file decoding, tokenization, splitting, windows, and
batching. The workflow helper chooses inputs and connects these operations.
It does not repeat their validation or splitting logic, seed global random
sources, or create an output directory.

Failures propagate to the caller. If text loading fails, loader construction
must not run. If construction fails, the helper lets the error propagate.
Later, the CLI boundary can translate expected errors into user-facing messages.

## Return a mock result when only forwarding matters

A test of orchestration does not need real loaders when it never iterates over
them. Configure the patched factory to return a known object:

```python
bundle = mocker.Mock(spec=DataLoaderBundle)
factory = mocker.patch(
    "gpt_2.workflow.create_data_loaders",
    autospec=True,
    return_value=bundle,
)
```

`return_value` is the result of calling the patched function, not the function
itself. `autospec=True` checks calls against the real factory signature.
The mock bundle is sufficient for this test because it is only passed through;
it does not validate batching or act as a functioning data loader.

After calling the helper, check the factory arguments and result identity:

```python
factory.assert_called_once_with(
    token_ids=token_ids,
    model_config=run.config.model,
    training_config=run.config.training,
    data_config=run.config.data,
)
assert result is bundle
```

`is` proves the helper returned the exact factory result. An equality assertion
could accept a different object with equal values.

Patch `gpt_2.workflow.load_text_token_ids` and
`gpt_2.workflow.create_data_loaders`, because those are the names the workflow
looks up. Existing dataset tests remain responsible for the factories' actual
behavior. A later integration test will connect the real components.

## Keep failure tests and fixtures purposeful

Use `side_effect=ValueError("token loading failure")` on the token-loading mock
and assert the loader factory was never called. This factory needs no configured
return value because successful construction must not be reached.

Use a loader-factory side effect to test construction-error propagation after
successful token loading. Verify both calls and the raised error.

Construct `PreparedPretrainRun` directly. When file loading is mocked, only
paths are needed; creating files does not contribute evidence to these tests.
Keep the fixture's resolved seed equal to its training configuration seed so
the fixture represents a consistent prepared run.

## Verification at completion

The assembled workflow passed the full 304-test suite. After fixture and test
cleanup, all nine workflow tests, Ruff linting, formatting, and ty also passed.
These counts describe this checkpoint rather than a permanent suite size.

## Further reading

- [Mock return values and side effects](https://docs.python.org/3/library/unittest.mock.html#the-mock-class)
- [Where to patch](https://docs.python.org/3/library/unittest.mock.html#where-to-patch)
- [Python identity comparisons](https://docs.python.org/3/reference/expressions.html#is-not)
- [Hermetic tests](06-hermetic-tests.md)

[Back to the engineering-notes index](../engineering-notes.md)
