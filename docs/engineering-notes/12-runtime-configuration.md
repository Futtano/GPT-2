# Validate typed configuration at runtime

Type annotations document intent and support static analysis, but dataclasses
do not enforce them at runtime. A field annotated as `int` can still receive a
float, string, or Boolean unless validation rejects it.

The model configuration uses a frozen dataclass:

```python
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool
```

`frozen=True` prevents accidental reassignment during an experiment. Validation
in `__post_init__` should proceed in this order:

1. Confirm dimensions are integers and not Booleans.
2. Confirm dimensions are positive.
3. Confirm dropout is numeric, finite, and within `[0.0, 1.0]`.
4. Confirm flags are actual Booleans.
5. Confirm `emb_dim` is divisible by `n_heads`.

Booleans need an explicit check because:

```python
isinstance(True, int)  # True
```

Range comparisons alone do not reject NaN. Use `math.isfinite` to reject NaN
and both infinities.

## Test one rule at a time

Start from a valid dictionary and override one field:

```python
values = VALID_CONFIG | {field: invalid_value}

with pytest.raises(ValueError, match=message):
    ModelConfig(**values)
```

Several invalid fields in one case allow the first exception to hide missing
validation for later fields. Include boundaries and cross-field cases
explicitly. Use `dataclasses.asdict` for resolved configuration serialization
and test the resulting field mapping.

## Normalize mappings at the boundary

Notebooks and configuration-file parsers naturally produce dictionaries,
while model internals benefit from typed attribute access. Support both by
normalizing once at the public boundary:

```python
def ensure_model_config(
    config: ModelConfig | Mapping[str, Any],
) -> ModelConfig:
    if isinstance(config, ModelConfig):
        return config

    return ModelConfig(**config)
```

Constructors can accept the compatibility union, convert immediately, and use
only typed attributes afterward:

```python
class GPTModel(nn.Module):
    def __init__(self, config: ModelConfig | Mapping[str, Any]):
        super().__init__()
        self.config = ensure_model_config(config)
        self.tok_emb = nn.Embedding(
            self.config.vocab_size,
            self.config.emb_dim,
        )
```

This keeps dictionary-based notebooks working while ensuring all model
construction passes through runtime validation. Tests should primarily pass
`ModelConfig` instances and retain one explicit mapping-compatibility test.

Give lower-level components only the settings they need. `FeedForward`, for
example, needs `emb_dim`, not the complete model configuration. Narrow inputs
reduce coupling and make components easier to test.

## Separate model and training configuration

Model architecture and training procedure change for different reasons, so
they belong in separate dataclasses. `ModelConfig` describes the network that
must be reconstructed when a checkpoint is loaded. `TrainingConfig` describes
how that network is optimized and evaluated:

```python
DeviceName = Literal["auto", "cpu", "cuda", "mps"]

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    eval_every_steps: int
    eval_batches: int
    checkpoint_every_steps: int
    seed: int
    device: DeviceName
```

Names should include their unit when a bare word would be ambiguous.
`eval_every_steps` makes it clear that the frequency counts optimizer steps,
not batches, epochs, examples, or tokens.

Use small validation functions for rules shared by multiple dataclasses. This
keeps each `__post_init__` readable and gives common concepts one consistent
meaning:

- counts such as batch size and epoch count are positive integers;
- seeds are non-negative integers, so seed zero remains valid;
- learning rates are finite real numbers strictly greater than zero;
- Booleans are rejected where integers are expected.

When validating an abstract `numbers.Real`, convert it to a concrete `float`
after the runtime type check. This gives static analyzers a concrete type for
comparisons. Catch `OverflowError` so a value too large to represent still
produces the validator's documented `ValueError`.

## Static constraints still need runtime checks

`Literal` tells a static type checker which strings are permitted, but Python
does not enforce it while constructing a dataclass. Keep a runtime collection
for validation:

```python
DeviceName = Literal["auto", "cpu", "cuda", "mps"]
ALLOWED_DEVICES = ("auto", "cpu", "cuda", "mps")

if self.device not in ALLOWED_DEVICES:
    raise ValueError(...)
```

The alias and tuple serve different consumers: `DeviceName` supports editors
and static analysis, while `ALLOWED_DEVICES` supports runtime validation.

Keep the configured device as a serializable string. Resolve `"auto"` to a
concrete `torch.device` at the execution boundary, where hardware availability
can be inspected. This keeps configuration independent of PyTorch runtime
state and easy to serialize with `dataclasses.asdict`.

## Further reading

- [Python documentation: `dataclasses`](https://docs.python.org/3/library/dataclasses.html)
- [Python documentation: `numbers`](https://docs.python.org/3/library/numbers.html)
- [Python documentation: `math.isfinite`](https://docs.python.org/3/library/math.html#math.isfinite)
- [Python documentation: `collections.abc.Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)
- [Python typing specification: literal types](https://typing.python.org/en/latest/spec/literal.html)

[Back to the engineering-notes index](../engineering-notes.md)
