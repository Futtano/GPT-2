# Handle PyTorch tensor copying explicitly

The test suite exposed a warning caused by copying an existing tensor with
`torch.tensor(source)`. The weight-assignment helper must accept NumPy arrays
and tensors while matching the destination model parameter.

The contract is:

- `left` supplies required shape, dtype, and device.
- `right` supplies values and may be a tensor or NumPy array.
- The result is a trainable `torch.nn.Parameter`.
- The result owns its storage; source mutation cannot change it.

The implementation pattern is:

```python
right_tensor = torch.as_tensor(
    right,
    dtype=left.dtype,
    device=left.device,
)
right_tensor = right_tensor.detach().clone()
return torch.nn.Parameter(right_tensor)
```

Each operation has a purpose:

- `as_tensor` accepts tensor and NumPy inputs and applies dtype/device
  conversion.
- `detach` removes the source autograd history.
- `clone` creates independent storage.
- `Parameter` creates trainable model state with `requires_grad=True`.

Tests should use different source and destination dtypes, verify device and
`requires_grad`, then mutate both source types and confirm the result remains
unchanged. Promote warnings to errors during verification:

```bash
uv run pytest -q -W error
```

## Further reading

- [PyTorch: `torch.as_tensor`](https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html)
- [PyTorch: `Tensor.detach`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)
- [PyTorch: `torch.clone`](https://docs.pytorch.org/docs/stable/generated/torch.clone.html)
- [PyTorch: `torch.nn.Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)

[Back to the engineering-notes index](../engineering-notes.md)
