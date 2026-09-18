# Keep unit-test models tiny

Unit tests usually verify shapes, finite values, loss behavior, and control
flow. They do not need the statistical capacity of GPT-2 124M.

Replacing repeated full-model construction with small configurations produced
representative improvements:

```text
Training call:         6.63s -> about 0.4s
GPT model test:        2-3s  -> about 0.03s
Warm full test suite: 13.35s -> 5.80s
```

Pre-training tests use the real GPT-2 tokenizer, so they retain the full
vocabulary while reducing depth and width:

```python
TINY_GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 32,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}
```

Pure model-shape tests create token IDs directly, allowing a small vocabulary:

```python
TEST_GPT_CONFIG = {
    "vocab_size": 128,
    "context_length": 16,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}
```

Use function-scoped model fixtures when tests train or mutate the model. A
shared instance can make outcomes depend on execution order.

Measure improvements:

```bash
uv run pytest -q --durations=10
```

## Further reading

- [pytest: Fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [PyTorch: Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

[Back to the engineering-notes index](../engineering-notes.md)
