# Build language-model windows lazily

Tokenization, splitting, and window construction are separate operations. The
production dataset accepts an already-split token sequence so no window can
cross from training data into validation or test data.

## Store the sequence once

Materializing every overlapping window duplicates most tokens when the stride
is smaller than the context length. Instead, `GPTTokenDataset` stores one
`torch.long` tensor and creates each input-target pair in `__getitem__`:

```python
start = index * stride
stop = start + context_length

inputs = token_ids[start:stop]
targets = token_ids[start + 1:stop + 1]
```

The two slices differ by one position, which defines next-token prediction.
This design keeps dataset storage proportional to the token count rather than
the number and width of overlapping windows.

## Calculate the number of complete windows

Each sample needs `context_length + 1` tokens: `context_length` input tokens
and one additional token at the end of the shifted target. After rejecting
short sequences, the number of complete windows is:

```python
num_windows = (
    (len(token_ids) - context_length - 1) // stride
) + 1
```

The subtraction finds the last valid starting offset, floor division counts
the complete stride movements, and the final addition includes the window at
offset zero.

Validate `context_length` and `stride` as positive, non-Boolean integers before
using them in arithmetic. A stride larger than the context length is still a
valid dataset operation; it intentionally leaves gaps between samples. A
higher-level workflow may reject gaps if complete token coverage is part of
its policy.

## Keep old entry points as adapters

Existing notebooks pass text and a tokenizer to `GPTDatasetV1`. Preserve that
interface as a thin compatibility wrapper that tokenizes the text and delegates
to `GPTTokenDataset`. The production pipeline can use `GPTTokenDataset`
directly after splitting, while older callers keep working.

Compatibility tests must compare the wrapper with the canonical dataset, not
two calls to the same object. Compare their lengths and every input-target
window. Fixed expected windows provide a separate test of the windowing policy
without repeating the implementation's formula.

## Further reading

- [PyTorch documentation: creating a custom dataset](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch documentation: `torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

[Back to the engineering-notes index](../engineering-notes.md)
