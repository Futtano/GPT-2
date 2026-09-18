# Assert expected failures directly

Use `xfail` for a known defect or unsupported behavior that remains unresolved.
Do not use it when the implementation already rejects invalid input as
designed.

An invalid attention configuration was originally included in a normal shape
test and marked as an expected failure. The clearer contract is a separate
test:

```python
with pytest.raises(
    ValueError,
    match="d_out must be divisible by num_heads",
):
    MultiHeadAttention(...)
```

For public input validation, prefer an explicit exception over `assert`:

```python
if d_out % num_heads != 0:
    raise ValueError("d_out must be divisible by num_heads")
```

Python may remove assertions when run with optimization. Assertions are better
suited to internal invariants; `ValueError` or `TypeError` communicates invalid
user input reliably.

Testing the message fragment matters because it verifies that users receive a
diagnostic tied to the invalid field rather than an unrelated exception.

## Further reading

- [pytest: How to use skip and xfail](https://docs.pytest.org/en/stable/how-to/skipping.html)
- [pytest: Assertions about expected exceptions](https://docs.pytest.org/en/stable/how-to/assert.html#assertions-about-expected-exceptions)
- [Python documentation: `assert`](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement)

[Back to the engineering-notes index](../engineering-notes.md)
