# Construct model and optimizer with shared parameter ownership

`create_pretrain_components` builds a real `GPTModel` from the prepared model
configuration, moves it to the resolved device, and then creates AdamW using
that model's parameters. The returned `PretrainComponents` names both objects.

## Move the model before constructing the optimizer

The construction order is explicit:

```python
model = GPTModel(cfg=run.config.model)
model.to(run.runtime.device)
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=run.config.training.learning_rate,
    weight_decay=0.0,
)
```

The optimizer must reference the parameters used by the model on its execution
device. Constructing it after device placement avoids holding references to
parameters that a device conversion may replace.

The learning rate comes from `TrainingConfig`. Weight decay is explicitly zero
for this initial workflow, rather than inheriting AdamW's nonzero default.
This is a fixed workflow policy, not a universal recommendation for training.
If weight decay becomes configurable, record it in validated configuration
and run artifacts.

The factory does not seed random sources, load data, or start optimization.
Seeding belongs to preparation and must occur before model initialization in
the assembled application.

## Inspect optimizer groups and compare identities

The optimizer's `param_groups` is a list of dictionaries. Each dictionary
contains `params`, `lr`, and other optimization settings. Flatten its parameter
lists for comparison:

```python
optimizer_parameters = [
    parameter
    for group in optimizer.param_groups
    for parameter in group["params"]
]
model_parameters = list(model.parameters())

assert len(optimizer_parameters) == len(model_parameters)
assert {id(p) for p in optimizer_parameters} == {id(p) for p in model_parameters}
```

The length and identity checks establish that the optimizer references exactly
the returned model's parameters. Tensor equality compares numerical contents,
which cannot prove shared object ownership and can produce ambiguous truth
values for multi-element tensors.

Assert learning rate and weight decay in every parameter group. A tiny real
CPU model makes this test inexpensive without mocking the behavior being
verified.

## Use a spy to observe a real operation

A model begins on CPU, so checking CPU parameter placement alone would still
pass if the factory omitted `model.to(...)`. A spy records the call while
executing the real method:

```python
move_to_device = mocker.spy(GPTModel, "to")
components = create_pretrain_components(prepared_run)
move_to_device.assert_called_once_with(
    components.model,
    prepared_run.runtime.device,
)
```

Because the spy is attached to the class method, its recorded arguments include
the model instance (`self`). The parameter-device assertions still verify the
real result. The CPU test does not prove accelerator execution or call ordering
on accelerator hardware.

## Share ordinary input records through fixtures

The `prepared_run` fixture constructs a real `PreparedPretrainRun` from the
existing configuration and runtime fixtures and temporary paths. Constructing
this dataclass does not call preparation or create files.

Reuse it in component and loader-orchestration tests to remove repeated setup.
This keeps inputs internally consistent while each test remains explicit about
the operation, dependency behavior, and assertions it owns.

Verification after adding the shared fixture and device spy: all ten workflow
tests, Ruff linting, formatting, and ty passed.

## Further reading

- [PyTorch Module device conversion](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [AdamW settings](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [pytest-mock spies](https://pytest-mock.readthedocs.io/en/latest/usage.html#spy)
- [pytest fixture reuse](https://docs.pytest.org/en/stable/how-to/fixtures.html)

[Back to the engineering-notes index](../engineering-notes.md)
