# Keep evaluation loaders separate from training randomness

`DataLoaderBundle` now contains four loaders over three disjoint splits:

| Loader | Dataset | Sampling |
| --- | --- | --- |
| `train` | Training split | Shuffled |
| `train_eval` | Same training dataset object | Sequential |
| `validation` | Validation split | Sequential |
| `test` | Test split | Sequential |

All four use the configured batch size and worker count and retain partial
final batches. The training-evaluation loader measures training loss without
reusing the shuffled optimization loader.

## Sharing data does not require sharing random state

Each loader has its own `torch.Generator`, explicitly seeded from the configured
seed. Separate generator objects can begin with identical states: drawing from
one still leaves the others unchanged. This establishes state isolation, not
statistical independence of identically seeded random sequences.

DataLoader iterator creation consumes a base seed. Consequently, even a
nonshuffled loader benefits from an explicit generator: evaluation should not
consume the global RNG used by model initialization or dropout. Our zero-worker
tests confirm that the evaluation loader advances its own generator while
leaving both training and global states unchanged.

## Check shared dataset identity and complete sample order

Use identity to verify the two training loaders share their dataset:

```python
assert bundle.train.dataset is bundle.train_eval.dataset
assert isinstance(bundle.train_eval.sampler, SequentialSampler)
```

For synthetic tokens increasing from zero, collect the first token of every
window across all batches. The expected starts are:

```python
expected = torch.arange(len(dataset), dtype=torch.long) * data_config.stride
```

Comparing the complete concatenated sequence catches reorderings, omissions,
and duplicates across batch boundaries. Checking only adjacent windows inside
each batch would miss errors between batches. Use the configured stride rather
than hardcoding an overlap offset.

## Snapshot states before iteration and compare afterward

`Generator.get_state()` and `torch.random.get_rng_state()` return state tensors.
They are snapshots of generator state, not random values to compare against
fixed expected numbers.

```python
train_before = train_generator.get_state().clone()

with torch.random.fork_rng(devices=[]):
    global_before = torch.random.get_rng_state().clone()
    batches = list(evaluation_loader)

    assert batches
    assert torch.equal(train_generator.get_state(), train_before)
    assert torch.equal(torch.random.get_rng_state(), global_before)
```

The test also snapshots the evaluation generator and verifies it changes.
This demonstrates that iteration consumed the intended generator. A separate
test checks that all four generator identities differ.

`fork_rng(devices=[])` restores global CPU state on exit, including when a
regression makes an assertion fail. It does not restore the explicitly created
loader generators, which are local to the test. The empty device list avoids
initializing or snapshotting CUDA RNGs.

Parameterize over `train_eval`, `validation`, and `test`, using `num_workers=0`
to keep this state-isolation test focused. This is not a test of random
transforms or other libraries' RNG state in worker processes.

## Extend existing tests when the bundle grows

Adding a loader to the dataclass is not enough. Include it in existing checks
for batch size, worker count, partial-batch retention, tensor shapes, and dtypes.
Keep the original split-disjointness checks: training evaluation intentionally
reuses training data, while validation and test data remain separate.

The full suite passed with 354 tests. The subsequent extension of the existing
loader checks was verified against the dataset suite. Ruff linting, formatting,
and ty were also checked for the completed change.

## Further reading

- [PyTorch DataLoader randomness](https://docs.pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading)
- [PyTorch Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html)
- [PyTorch fork_rng](https://docs.pytorch.org/docs/stable/random.html#torch.random.fork_rng)
- [Reproducible split loaders](16-reproducible-data-loaders.md)
- [Typed training loop](27-typed-training-progress-and-scheduling.md)

[Back to the engineering-notes index](../engineering-notes.md)
