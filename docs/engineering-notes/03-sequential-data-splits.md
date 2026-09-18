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

## Further reading

- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: Time-series cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

[Back to the engineering-notes index](../engineering-notes.md)
