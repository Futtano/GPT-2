# Write assertions that can fail meaningfully

A passing test provides evidence only when its assertions can detect the defect
they claim to cover.

For a tensor shaped `(1, 0)`, `len(tensor)` is `1` because `len` reports the
first dimension. Test empty token content with:

```python
assert token_ids.shape == (1, 0)
assert token_ids.numel() == 0
```

Adjacent Python string literals concatenate automatically:

```python
[
    "Hello",
    "",  # this comma is required
]
```

Without the comma, an intended empty-string parameter silently disappears.

Other useful rules:

- Prefer `torch.equal` for exact tensor equality and `torch.allclose` for
  floating-point comparisons.
- Check finiteness when infinity would satisfy a simple non-negative check.
- Avoid vacuous conditions such as `len(values) >= 0`.
- Verify state changes when testing training, not only return types.
- Test exception type and a meaningful message fragment.
- Make test inputs challenge conversions; matching default dtypes prove less
  than deliberately different source and destination dtypes.
- When testing storage ownership, mutate the source after creating the result.

Read a test by asking: “What incorrect implementation would still pass this?”
Then strengthen the input or assertion to distinguish that implementation.

## Further reading

- [pytest: Writing and reporting assertions](https://docs.pytest.org/en/stable/how-to/assert.html)
- [PyTorch: Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

[Back to the engineering-notes index](../engineering-notes.md)
