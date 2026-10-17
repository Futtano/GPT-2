# Assemble the workflow and test stage ordering

`run_pretraining` in `workflow.py` connects the tested application boundaries:

```text
prepare arguments and runtime
    -> load tokens and construct loaders
    -> construct model and optimizer
    -> train and return TrainingResult
```

Constructing loaders before the model allows unusable input data to fail before
model allocation. The function forwards the resolved device and training
configuration and passes `loaders.train_eval` for training metrics. The held-out
test loader is not passed to training.

The result remains in memory. This milestone does not save metrics, select and
persist a model, expose the installed command, or provide resume behavior.

## Distinguish a patched function from its return value

Configure the loader factory like this:

```python
mock_loaders = mocker.patch(
    "gpt_2.workflow.create_pretrain_loaders",
    autospec=True,
    return_value=loaders,
)
```

`mock_loaders` represents the function. Calling it returns `loaders`, the
fixture containing the loader objects. The training assertion must therefore
expect `loaders.train`, `loaders.train_eval`, and `loaders.validation`.

Accessing `mock_loaders.train` instead refers to an attribute of the factory
mock, not an attribute of its returned bundle. Without a restrictive spec,
mock attributes can be created automatically and hide this distinction.

The success test checks each stage's arguments with
`assert_called_once_with(...)` and checks `result is expected_results` to prove
the exact training result is returned.

## Record calls across multiple functions

Individual mock assertions do not establish relative order between different
functions. Attach the patched functions to a parent mock before invoking the
workflow:

```python
calls = mocker.Mock()
calls.attach_mock(mock_prepare, "prepare")
calls.attach_mock(mock_loaders, "loaders")
calls.attach_mock(mock_components, "components")
calls.attach_mock(mock_train, "train")
```

After execution, the test extracts the names:

```python
assert [call[0] for call in calls.mock_calls] == [
    "prepare", "loaders", "components", "train",
]
```

Entries in `mock_calls` are call objects that unpack as
`(name, args, kwargs)`. The name is the attached alias, positional arguments
are a tuple, and keyword arguments are a dictionary. Name-list equality checks
the entire recorded sequence while individual assertions verify arguments.

## Inject failures at each boundary

An exception supplied as `side_effect` raises when the patched function is
called. No real missing input file is needed to exercise preparation failure:

```python
mocker.patch(
    "gpt_2.workflow.prepare_pretrain_run",
    autospec=True,
    side_effect=FileNotFoundError("Configuration file not found"),
)
```

The failure tests cover these recorded prefixes:

| Failing stage | Calls reached | Later stages |
| --- | --- | --- |
| Preparation | prepare | Not called |
| Loaders | prepare, loaders | Not called |
| Components | prepare, loaders, components | Training not called |
| Training | prepare, loaders, components, train | Error propagates |

These tests verify error propagation and stopping subsequent work. They do not
promise rollback of earlier side effects: preparation may already have created
an output directory before a later failure.

## Use purpose-specific fixtures

The loader-bundle fixture contains four distinct `Mock(spec=DataLoader)`
objects. The assembly test forwards them without iteration, so actual datasets
and loaders would add unnecessary setup. `DataLoader(Dataset())` would also
wrap an unimplemented base dataset, which is unsuitable if iteration is later
introduced.

The synthetic result fixture uses distinct progress records for epochs one
and two. Its final record agrees with the result's final step and token count.
It is a configured mock result, not the outcome of real training in this test.

The shared tiny model fixture now lives in `tests/unit/gpt_2/conftest.py`.
pytest discovers fixtures there for test modules beneath that directory;
request them by parameter name rather than importing the fixture function.

## Verification and remaining integration work

The workflow assembly is covered by mocks, alongside the existing real model,
loader, and training tests. The next checkpoint will execute a tiny CPU run
through the real boundaries in one integration test. Unit assembly tests alone
cannot establish end-to-end compatibility.

## Further reading

- [Mock attach_mock](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.attach_mock)
- [Mock call objects](https://docs.python.org/3/library/unittest.mock.html#calls-as-tuples)
- [pytest fixture discovery](https://docs.pytest.org/en/stable/reference/fixtures.html)
- [Workflow composition](23-workflow-composition-and-mock-results.md)
- [Preparation side-effect boundaries](21-resolved-runtime-and-preparation.md)

[Back to the engineering-notes index](../engineering-notes.md)
