# Resolve runtime state at the preparation boundary

Configuration records what a user requests. Runtime state records the effective
settings used for execution. For example, `TrainingConfig.device` can remain
`"auto"` while the prepared run records `torch.device("cpu")`.

## Keep requested and effective values explicit

`runtime.py` defines a small frozen dataclass:

```python
@dataclass(frozen=True)
class ResolvedRuntime:
    device: torch.device
    seed: int
```

`PreparedPretrainRun` carries operational arguments, validated `RunConfig`, and
this resolved state. Downstream code can use `run.runtime.device` without
repeating hardware detection. The seed remains visible alongside the effective
device, which will also help when writing run metadata.

Automatic device selection prefers CUDA, then MPS, then CPU. Explicit CPU skips
accelerator probes. An explicitly requested unavailable accelerator raises an
error. Unit tests mock hardware checks so these behaviors are independent of
the developer's machine.

## Validate before applying side effects

`prepare_pretrain_run` currently performs these operations in order:

1. Verify that configuration and input paths identify files.
2. Load and validate the TOML configuration.
3. Reject an existing output path.
4. Resolve the requested device.
5. Seed Python, NumPy, and PyTorch random sources.
6. Create a fresh output directory with `exist_ok=False`.
7. Return the prepared run.

This order prevents known input, configuration, overwrite, and device errors
from changing global random state or creating output artifacts. The early
existence check avoids unnecessary runtime initialization; the final exclusive
directory creation still protects against another process creating the path
between those operations.

Preparation is not a transaction. Directory creation can fail after seeding
because of permissions or a concurrent creator. Global random state is not
rolled back in that case. Tests should assert the guarantees provided by each
failure path rather than claim all side effects are always reversible.

## Seed sources once before constructing training objects

`seed_random_sources` calls `random.seed`, `np.random.seed`, and
`torch.manual_seed`. The configuration accepts integer seeds from zero through
`2**32 - 1`, matching the common supported range for these calls. Booleans are
rejected even though Python treats them as integer subclasses.

The training loader retains its separately seeded `torch.Generator` for
shuffling. Global PyTorch seeding controls operations such as model parameter
initialization and dropout. A seed is not a promise of identical results across
different hardware, library versions, or nondeterministic operations.

## Test orchestration without touching global state

Workflow tests patch the names used by `workflow.py`:

```python
resolve = mocker.patch(
    "gpt_2.workflow.resolve_device",
    autospec=True,
    return_value=torch.device("cpu"),
)
seed = mocker.patch("gpt_2.workflow.seed_random_sources", autospec=True)
```

Assert the resolved state, arguments passed to each dependency, and absence of
later calls on failure. With `autospec=True`, mock assertions match arguments
against the original function signature, so equivalent positional and keyword
calls can match.

Test seed bounds through `TrainingConfig`. Testing an invalid seed by calling
the real runtime helper changes Python's RNG before NumPy rejects the value.
The mocked helper test checks that each seeder receives the seed once without
changing the test process's global RNG state.

## Further reading

- [Python dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Mock autospeccing](https://docs.python.org/3/library/unittest.mock.html#autospeccing)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [NumPy RandomState seed range](https://numpy.org/doc/stable/reference/random/legacy.html)

[Back to the engineering-notes index](../engineering-notes.md)
