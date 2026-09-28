# Test typed boundaries without hiding errors

Runtime validation and static type checking answer different questions. A type
checker asks whether ordinary program code respects declared interfaces.
Runtime-validation tests deliberately violate those interfaces to prove that
bad external input is rejected safely.

Tests should make that distinction explicit so real typing mistakes remain
visible.

## Build valid fixtures with precise types

A heterogeneous dictionary usually loses the relationship between each key
and its value type:

~~~python
values = {
    "vocab_size": 128,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

ModelConfig(**values)
~~~

Without a more precise annotation, a checker may infer every value as a union
such as int, float, or bool. It then cannot prove that qkv_bias receives the
Boolean member of that union.

For ordinary valid fixtures, prefer direct construction:

~~~python
config = ModelConfig(
    vocab_size=128,
    context_length=32,
    emb_dim=16,
    n_heads=4,
    n_layers=1,
    drop_rate=0.0,
    qkv_bias=False,
)
~~~

Direct construction is readable and lets the checker validate every field. If
a reusable mapping is part of the behavior under test, describe its keys with
TypedDict instead of weakening it to dict[str, Any].

## Mark deliberate static violations locally

A negative runtime test may intentionally pass a string where an integer is
declared:

~~~python
values = VALID_MODEL_CONFIG | {"vocab_size": "invalid"}

with pytest.raises(ValueError, match="vocab_size"):
    ModelConfig(**values)  # ty: ignore[invalid-argument-type]
~~~

A rule-specific suppression on the exact call is honest: the call is
statically invalid by design, and the test is checking its runtime outcome.

Use the narrowest suitable mechanism:

1. Write valid fixtures with explicit, correctly typed values.
2. Use TypedDict for valid heterogeneous mappings.
3. Add a rule-specific line suppression for one intentional violation.
4. Use a file-specific override only when a whole boundary-test module
   deliberately contradicts annotations.

Do not introduce Any or a misleading cast merely to make diagnostics
disappear. Both approaches can conceal accidental mistakes beyond the intended
negative test. Avoid blanket ignore comments because they also suppress
unrelated diagnostics on the same line.

## Mock at the lookup site

When a module imports a dependency directly:

~~~python
from gpt_2.config import load_run_config
~~~

the consuming module stores its own reference. A workflow test must therefore
patch the name used by that module:

~~~python
mock_load_run_config = mocker.patch(
    "gpt_2.workflow.load_run_config",
    autospec=True,
    return_value=run_config,
)
~~~

Patching gpt_2.config.load_run_config would replace the original definition,
but workflow.py would continue using the reference it imported earlier.
Patch where the code under test looks up the name.

pytest-mock restores patched objects automatically after each test. This
avoids context-manager nesting and prevents one test's replacement from
leaking into another test.

## Preserve the dependency contract with autospec

autospec=True constructs the mock from the real callable's signature. If
production code later calls the dependency with incompatible arguments, the
test fails rather than accepting any call silently.

Use return_value to model success:

~~~python
mock_load_run_config = mocker.patch(
    "gpt_2.workflow.load_run_config",
    autospec=True,
    return_value=run_config,
)
~~~

Use side_effect to model failure:

~~~python
mock_load_run_config = mocker.patch(
    "gpt_2.workflow.load_run_config",
    autospec=True,
    side_effect=ValueError("Invalid configuration"),
)
~~~

The latter lets a workflow test verify that configuration failure propagates
and that later side effects, such as creating an output directory, do not
occur.

## Assert both outcomes and interactions

An output assertion checks observable behavior:

~~~python
assert not output_dir.exists()
~~~

A mock interaction assertion checks orchestration:

~~~python
mock_load_run_config.assert_called_once_with(config_file)
mock_load_run_config.assert_not_called()
~~~

Use interaction assertions only where call order or dependency use is part of
the contract. The successful preparation test should prove that the loader
receives the validated path. Missing-file tests should prove that validation
short-circuits before the loader runs.

## Mock only the boundary under test

The workflow tests keep real pathlib operations under pytest's tmp_path while
mocking TOML loading. This division is useful because:

- path validation and directory creation are the behavior under test;
- TOML parsing already has dedicated tests;
- no network, global filesystem, or persistent user state is involved;
- failures identify the responsible layer clearly.

Mocking pathlib itself would mostly test calls to mocks and could miss actual
filesystem behavior. Calling the real TOML parser in every workflow unit test
would duplicate parser coverage and make fixtures unnecessarily large.

## Further reading

- [pytest-mock usage](https://pytest-mock.readthedocs.io/en/latest/usage.html)
- [Python documentation: where to patch](https://docs.python.org/3/library/unittest.mock.html#where-to-patch)
- [ty suppression comments](https://docs.astral.sh/ty/suppression/)
- [Python typing specification: TypedDict](https://typing.python.org/en/latest/spec/typeddict.html)
- [pytest temporary directories](https://docs.pytest.org/en/stable/how-to/tmp_path.html)

[Back to the engineering-notes index](../engineering-notes.md)
