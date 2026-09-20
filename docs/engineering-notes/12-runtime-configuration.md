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

## Further reading

- [Python documentation: `dataclasses`](https://docs.python.org/3/library/dataclasses.html)
- [Python documentation: `numbers`](https://docs.python.org/3/library/numbers.html)
- [Python documentation: `math.isfinite`](https://docs.python.org/3/library/math.html#math.isfinite)
- [Python documentation: `collections.abc.Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

[Back to the engineering-notes index](../engineering-notes.md)
