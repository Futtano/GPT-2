# Average evaluation loss over tokens

`calc_loss_batch` returns mean cross-entropy over the target positions in one
batch. `calc_loss_loader` combines those means using their target-token counts.
This matters when the loader retains a smaller final batch.

## Weight batch means by the observations they represent

If one batch contains eight target tokens with mean loss 2 and another contains
two with mean loss 10, the loader-level loss is:

```text
(8 * 2 + 2 * 10) / (8 + 2) = 3.6
```

The unweighted mean, `(2 + 10) / 2 = 6`, gives the smaller batch too much
influence. The implementation accumulates:

```python
total_loss += loss.item() * target_batch.numel()
total_tokens += target_batch.numel()
```

Divide by `total_tokens` after evaluation. For targets shaped
`(batch_size, context_length)`, `numel()` counts all next-token predictions.
This denominator assumes the current loss has no masked or ignored targets.
If padding or ignored targets are added later, count only the positions included
in the loss reduction.

## Bound evaluation and reject invalid limits

`num_batches=None` evaluates the whole loader. A supplied limit must be a
positive integer, with booleans rejected explicitly. A type annotation does not
perform runtime validation, and `bool` is an integer subclass in Python.

Use `islice(data_loader, num_batches)` to evaluate a prefix. A limit larger than
the loader length naturally stops at exhaustion. Unlike an enumerate loop
that breaks after receiving the next batch, this avoids requesting an extra
batch from the iterator. DataLoader worker prefetching can still occur.

## Distinguish no batches from batches with no tokens

The tests construct a `TensorDataset` from inputs and targets with these shapes:

| Tensor shape | Dataset samples | Loader behavior | Failure path |
| --- | --- | --- | --- |
| `(0, 0)` | Zero | No batches | Empty-loader check |
| `(1, 0)` | One sample with zero token positions | One zero-token batch | Final token-count check |

Both tensors have rank two. A zero-sized axis is different from rank zero:
`torch.tensor(3.0)` is a scalar with shape `()` and `ndim == 0`.

Skip zero-token batches before calling the batch-loss function. If no target
tokens were evaluated, raise `ValueError` rather than divide by zero or return
NaN. Downstream metrics and model selection need a real evaluation result or
an explicit failure.

## Make the regression test distinguish correct and incorrect formulas

Use five synthetic samples with two target tokens each and a loader with
`batch_size=4, drop_last=False`. The two batches then contain eight and two
target tokens. Configure the batch-loss mock:

```python
batch_loss = mocker.patch(
    "gpt_2.pretrain.calc_loss_batch",
    side_effect=[torch.tensor(2.0), torch.tensor(10.0)],
)
```

Assert the complete result is approximately 3.6. With `num_batches=1`, assert
the result is 2 and the loss function was called once. With a limit exceeding
the two batches, assert the complete weighted result again.

Returning 1 for every batch cannot distinguish weighted and unweighted means.
Keep these tests synthetic and separate from the small shared training fixture
so metric coverage does not multiply real training work.

## Understand mock side effects

`return_value` supplies the same result on each call. `side_effect` can instead:

- return successive values from an iterable; exhaustion raises `StopIteration`;
- raise a supplied exception;
- call a supplied function with the patched function's arguments.

Call recording remains available in all three cases. An exception side effect
tests failure propagation; sequential loss values test batch aggregation.

Tests intentionally passing floats, booleans, or strings to an integer-annotated
parameter use a local `ty: ignore[invalid-argument-type]` at that call. This keeps
the invalid-input test explicit without weakening checking throughout the file.

## Scope of this change

The loader-loss helper retains its existing model-mode and gradient behavior.
Its evaluation caller owns `model.eval()` and `torch.no_grad()`. The next task
is to restore the caller's original model mode even when evaluation raises.

Verification at this checkpoint: 316 tests passed in 7.48 seconds, with Ruff
linting, formatting, ty, and diff checks passing.

## Further reading

- [Python itertools.islice](https://docs.python.org/3/library/itertools.html#itertools.islice)
- [Mock side_effect](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.side_effect)
- [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [PyTorch Tensor.numel](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numel.html)

[Back to the engineering-notes index](../engineering-notes.md)
