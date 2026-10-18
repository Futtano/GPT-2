# Restore model mode after evaluation

`evaluate_model` computes training and validation losses without changing the
caller's original training-mode flag. Its signature returns
`tuple[float, float]`, with training loss first and validation loss second.

## Model mode and gradient recording are separate controls

`model.eval()` changes behavior in modules such as dropout. It does not disable
gradient recording. `torch.no_grad()` disables gradient recording for the work
inside its context, but does not switch the model to evaluation mode.

Evaluation therefore uses both controls. The model-mode change is temporary:

```python
was_training = model.training
model.eval()
try:
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
finally:
    model.train(was_training)
```

An unconditional `model.train()` would incorrectly change a model that entered
in evaluation mode. Restoring only after successful loss calculation would
leave a training model in evaluation mode when either loss calculation raises.

The `finally` block runs on success and failure. It restores the mode without
suppressing the original loss error. The `no_grad` context also restores the
caller's gradient-recording state when it exits.

This implementation restores the top-level mode through `model.train(...)`,
which applies that mode recursively. It fits the current GPT model's uniform
mode policy; it does not preserve independently configured mixed submodule
training flags.

## Inspect temporary state while the call is executing

Use a callable mock side effect to observe the model inside evaluation:

```python
def fake_loss(data_loader, model, device, num_batches=None):
    assert model.training is False
    assert torch.is_grad_enabled() is False
    return 2.0


mocker.patch(
    "gpt_2.pretrain.calc_loss_loader",
    autospec=True,
    side_effect=fake_loss,
)
```

Recorded mock arguments hold references to the model, not historical snapshots
of its attributes. Checking `model.training` through `call_args_list` after
evaluation would observe its restored state. The callable side effect sees
the temporary state at the relevant point in execution.

Return a float from the callback so it matches the replaced helper's contract.

## Cover both starting modes and both failure positions

Parameterize success tests over initial training and evaluation modes. For
failure tests, also parameterize which loss call raises:

```python
error = ValueError("evaluation failed")
side_effect = [2.0, error] if fail_on_validation else error
```

The first form returns a training loss and then raises during validation. The
second raises on the first call. In all four combinations of starting mode
and failure position, assert the error propagates and the original mode is
restored.

## Distinguish train and validation dependencies

Use two distinct loader mocks in forwarding tests. Passing the same object for
both loaders cannot detect accidentally computing training loss twice.

Assert the calls in order using `assert_has_calls(..., any_order=False)` and
also assert `call_count == 2`. Ordered call assertions alone can allow extra
calls before or after the expected sequence.

Verify the model, loader, device, and evaluation-batch limit on each call.
Separate sequential mock results, such as `[2.0, 3.0]`, establish which loss is
returned in each tuple position.

Passing a mocked CUDA device in a forwarding test does not require GPU hardware
because the actual loss calculation is replaced. It checks argument forwarding,
not accelerator execution.

## Completion checkpoint

All 330 tests passed with warnings treated as errors. Ruff linting, formatting,
and ty passed across the configured source and test scope. Evaluation is ready
to be called from the training loop.

## Further reading

- [PyTorch Module.eval and train](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)
- [Python finally clauses](https://docs.python.org/3/tutorial/errors.html#defining-clean-up-actions)
- [Mock assert_has_calls](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.assert_has_calls)

[Back to the engineering-notes index](../engineering-notes.md)
