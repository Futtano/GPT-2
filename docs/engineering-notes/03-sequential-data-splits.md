# Split sequential data before creating windows

Overlapping language-model windows can leak nearly identical samples across
dataset splits. Split the original sequence first, then create windows within
each split.

Given 1,000 tokens:

```text
Training:   tokens   0-799
Validation: tokens 800-899
Test:       tokens 900-999
```

Create samples independently:

```python
train_windows = make_windows(tokens[:800])
val_windows = make_windows(tokens[800:900])
test_windows = make_windows(tokens[900:])
```

If windows are created first and randomly divided afterward, these samples
could enter different splits:

```text
Training sample:   tokens 0-99
Validation sample: tokens 1-100
```

They share 99 tokens. Validation loss would partly measure memorization of
training content and overstate generalization.

Each split has one role:

- **Training:** updates model parameters.
- **Validation:** selects checkpoints and informs development decisions.
- **Test:** provides final evaluation after model selection.

Record the source-data identity and exact split offsets with each run. This
allows later runs to reconstruct the same evaluation boundary.

## Use one deterministic rounding policy

The package first tokenizes the complete source and then splits the token ID
sequence. For a sequence of `n_tokens`, it computes:

```python
train_size = int(n_tokens * train_fraction)
validation_size = int(n_tokens * validation_fraction)

train_end = train_size
validation_end = train_end + validation_size
```

The training and validation sizes therefore round down, and the test split
receives every remaining token. This gives three ordered, non-overlapping
slices whose concatenation reconstructs the original sequence exactly.

The split function accepts a general `Sequence[int]` and returns three lists.
Normalizing the output type gives callers one stable contract whether the
input was a list, tuple, or range. It also ensures a returned list can be
modified without mutating a list supplied by the caller.

Configuration validation can prove that each requested fraction is positive,
but only the splitting function knows the actual token count. It rejects a
computed split when rounding leaves that split empty and reports the total,
training, validation, and test token counts in the exception.

Tests use a fixed sequence with known boundaries to document the rounding
policy. They also verify exact reconstruction, input independence, support for
different sequence implementations, and failures for empty or undersized
inputs.

## Further reading

- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: Time-series cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

[Back to the engineering-notes index](../engineering-notes.md)
