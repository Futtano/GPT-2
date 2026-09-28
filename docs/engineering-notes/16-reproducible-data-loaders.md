# Build reproducible loaders from disjoint token splits

The production loader factory composes `ModelConfig`, `TrainingConfig`, and
`DataConfig`. It splits one token sequence, verifies that every split can form
at least one complete window, constructs three independent
`GPTTokenDataset` instances, and returns them in a named `DataLoaderBundle`.

Named fields make call sites clearer than positional tuple unpacking:

```python
loaders.train
loaders.validation
loaders.test
```

## Give each split the correct sampling policy

Training uses random sampling because changing sample order between epochs is
useful for optimization. Validation and test use sequential sampling so their
order is stable and easy to inspect.

The training loader receives a dedicated seeded `torch.Generator`. Two loader
bundles constructed with the same seed therefore produce the same first-epoch
training order. The generator state advances during iteration, so later epochs
remain deterministic while receiving new orders.

All three loaders retain their final partial batch. GPT models do not require a
fixed batch dimension, and keeping `drop_last=False` prevents small datasets
from silently losing examples or producing no batches.

## Test behavior independently

A test should not rebuild the implementation and compare the two copies. The
same mistake can then exist on both sides. Prefer observable invariants:

- stored dataset tokens equal the exact expected source slices;
- training uses `RandomSampler` while evaluation uses `SequentialSampler`;
- the number of loaded samples equals the dataset length;
- the final batch has the exact expected remainder size;
- two independently constructed bundles produce the same seeded order;
- known validation and test inputs appear in a fixed expected order;
- every batch has the configured context width and `torch.long` dtype.

Use enough windows and more than one training batch in shuffle tests. A loader
with only one window or one batch can appear reproducible even when its random
sampling configuration is incorrect.

## Validate after computing real split sizes

Positive split fractions do not guarantee usable datasets. Once the actual
token counts are known, the factory requires at least
`context_length + 1` tokens in every split and names the failing split in its
exception. This keeps configuration validation separate from validation that
depends on a particular dataset.

## Further reading

- [PyTorch documentation: data loading order and samplers](https://docs.pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)

[Back to the engineering-notes index](../engineering-notes.md)
