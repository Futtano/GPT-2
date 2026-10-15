# Track training progress and evaluate the final model

`train_model` in `pretrain.py` runs optimization using `TrainingConfig` and
returns a typed `TrainingResult`. The educational `train_model_simple` remains
available. The new loop does not print samples, write artifacts, persist
checkpoints, or use the held-out test split.

## Count completed updates and processed target positions

Both `global_step` and `tokens_seen` start at zero. For each training batch:

1. Clear gradients.
2. Compute the differentiable batch loss.
3. Backpropagate.
4. Perform the optimizer update.
5. Increment the step count and add `target_batch.numel()` to the token count.

A step therefore means a completed optimizer update. Token counts include
partial batches and repeated target positions from overlapping windows; they
are not counts of unique source tokens. Evaluation does not increment them.

The tests use nine samples, each with four target positions, and batch size
four. Each epoch has batches of four, four, and one samples: three updates and
36 target tokens. Two epochs produce six updates and 72 target tokens.

For this TensorDataset fixture, calculate expected training tokens without
iterating the shuffled loader:

```python
targets = training_dataset.tensors[1]
expected_tokens = targets.numel() * config.num_epochs
```

This expectation assumes `drop_last=False` and one visit per sample per epoch.
Different sampler policies require a different expectation.

## Evaluate after scheduled updates and once at the end

Run periodic evaluation after an update when:

```python
global_step % config.eval_every_steps == 0
```

After all epochs, evaluate again only if the final update was not already
evaluated. This final block belongs outside the epoch loop. Evaluating after
every epoch would add observations beyond the selected schedule.

For six updates, the regression tests cover:

| Evaluation interval | Recorded steps |
| --- | --- |
| 2 | 2, 4, 6 |
| 4 | 4, 6 |
| 10 | 6 |

The final evaluation measures the model after its last update. It does not
evaluate only the training batches since the previous evaluation. Each call
uses the separate training-evaluation and validation loaders, limited by
`config.eval_batches` as usual.

## Return records with consistent counters

`EvaluationRecord` contains a one-based epoch number, completed global step,
training-token count, train loss, and validation loss. `TrainingResult` contains
the completed epoch count, final counters, and a tuple of those records.

With the synthetic fixtures and an interval of four, the first record belongs
to epoch two, step four, at 52 tokens: 36 from epoch one plus 16 from the first
batch of epoch two. The final record belongs to epoch two, step six, at 72
tokens. Distinct mocked losses verify that values reach the correct fields.

The records describe progress and metrics. They do not contain model weights,
optimizer state, or enough information to resume training; checkpoint support
is a later milestone.

## Use real updates, spies, and mocks for different evidence

A spy on `optimizer.step` records calls while executing real updates:

```python
step_spy = mocker.spy(optimizer, "step")
```

Check its count against the number of training batches times the epoch count.
Also clone initial parameters with `detach().clone()` and verify at least one
parameter changed. Calling the optimizer alone does not prove a useful update
occurred. The test starts the model in evaluation mode to exercise the loop's
switch back to training mode.

Scheduling tests mock `evaluate_model` with known float losses. Parameterize
the interval and create a configuration using `dataclasses.replace` because
TrainingConfig is frozen. Assert both the recorded step list and evaluation
call count, including no duplicate final evaluation.

An empty training loader must fail before gradient clearing, loss calculation,
optimizer updates, or evaluation. Its fixture is a typed empty TensorDataset,
which agrees with DataLoader's static annotations.

## Keep evaluation independent from optimization sampling

`train_eval_loader` is a separate loader over the same training dataset.
Passing the shuffled optimization loader into evaluation would consume its
generator and can change subsequent training order. The test fixtures use
nonshuffled evaluation loaders and independent explicit generators.

Passing different text to the training-evaluation loader would measure a
different dataset, not training loss. The fixtures share the training dataset
object and use distinct data for validation. Four-token synthetic windows also
avoid the legacy text-loader default of 256 tokens, which exceeds the tiny
model's 32-token context.

At this checkpoint the application factory still needed to provide the new
training-evaluation loader. This is now covered by
[evaluation-loader RNG isolation](28-evaluation-loader-rng-isolation.md),
including explicit evaluation generators:
DataLoader can draw an iterator base seed even when shuffling is disabled, so
evaluation should not consume the global RNG used by training dropout.

## Completion checkpoint

The full suite passed with 338 tests. The final empty-fixture typing correction
was additionally verified by its focused test. Ruff linting, formatting, ty,
and diff checks passed.

## Further reading

- [pytest-mock spies](https://pytest-mock.readthedocs.io/en/latest/usage.html#spy)
- [Python dataclasses.replace](https://docs.python.org/3/library/dataclasses.html#dataclasses.replace)
- [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [Token-weighted evaluation](25-token-weighted-loss-and-empty-evaluation.md)
- [Mode-preserving evaluation](26-mode-preserving-evaluation.md)

[Back to the engineering-notes index](../engineering-notes.md)
