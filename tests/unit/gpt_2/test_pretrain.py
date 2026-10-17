from dataclasses import replace

import numpy as np
import pytest
import tiktoken
import torch
from pytest_mock import MockerFixture
from torch.utils.data import DataLoader, TensorDataset

from gpt_2.config import TrainingConfig
from gpt_2.dataset import create_data_loader_v1
from gpt_2.pretrain import (
    EvaluationRecord,
    TrainingResult,
    assign,
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    generate,
    softmax_with_temperature,
    text_to_token_ids,
    token_ids_to_text,
    train_model,
    train_model_simple,
)

TINY_TRAINING_CONFIG = TrainingConfig(
    batch_size=4,
    learning_rate=1e-3,
    num_epochs=2,
    eval_every_steps=2,
    eval_batches=2,
    checkpoint_every_steps=2,
    seed=123,
    device="cpu",
)


@pytest.mark.parametrize(
    "input",
    [
        "My name is Daniele",
        "Hello, foo!",
        "",
    ],
)
def test_text_to_token_ids(input):
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = text_to_token_ids(input, tokenizer)

    # Must return a tensor
    assert isinstance(token_ids, torch.Tensor)
    # Must be 2-dimensional with first dimension equal to 1
    assert token_ids.shape[0] == 1

    # if empty string, an empty list must be returned
    if len(input) == 0:
        assert token_ids.numel() == 0
        assert token_ids.shape == (1, 0)
    else:  # each item must be an integer index
        for el in token_ids[0]:
            assert isinstance(el.item(), int)


@pytest.mark.parametrize(
    "input",
    [
        torch.tensor([]).unsqueeze(0),
        torch.tensor([1, 2, 3, 4]).unsqueeze(0),
        torch.tensor([1, 4, 5, 7, 50256]).unsqueeze(0),
    ],
)
def test_token_ids_to_text(input):
    tokenizer = tiktoken.get_encoding("gpt2")
    text = token_ids_to_text(input, tokenizer)

    # Must return a string
    assert isinstance(text, str)

    # if empty tensor, an empty string must be returned
    if len(input) == 0:
        assert len(text) == 0


@pytest.fixture
def text():
    return "Hello I am Daniele"


@pytest.fixture
def batches(text):
    context_length = 3
    stride = 1
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(text)
    input_batch, target_batch = [], []
    for i in range(0, len(token_ids) - context_length, stride):
        input_batch.append(torch.tensor(token_ids[i : i + context_length]))
        target_batch.append(torch.tensor(token_ids[i + 1 : i + context_length + 1]))

    input_batch = torch.stack(input_batch, dim=0)
    target_batch = torch.stack(target_batch, dim=0)

    return input_batch, target_batch


def test_calc_loss_batch(batches, model):
    input_batch, target_batch = batches
    loss = calc_loss_batch(input_batch, target_batch, model, torch.device("cpu"))
    assert loss.item() >= 0


@pytest.fixture
def dataloader():
    return create_data_loader_v1(
        "Hello from Daniele!",
        batch_size=2,
        max_length=4,
        stride=1,
        shuffle=False,
        num_workers=0,
    )


@pytest.fixture
def uneven_dataloader() -> DataLoader:
    # Four samples followed by one: eight target tokens, then two.
    tokens = torch.zeros((5, 2), dtype=torch.long)
    return DataLoader(
        TensorDataset(tokens, tokens.clone()),
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )


@pytest.mark.parametrize("num_batches", [None, 3])
def test_calc_loss_loader_produces_token_weighted_loss(
    uneven_dataloader: DataLoader,
    num_batches: int | None,
    mocker: MockerFixture,
):
    batch_loss = mocker.patch(
        "gpt_2.pretrain.calc_loss_batch",
        side_effect=[torch.tensor(2.0), torch.tensor(10.0)],
    )

    loss = calc_loss_loader(
        uneven_dataloader,
        torch.nn.Identity(),
        torch.device("cpu"),
        num_batches=num_batches,
    )

    assert batch_loss.call_count == 2
    assert loss == pytest.approx(3.6)


def test_calc_loss_loader_excludes_batches(
    uneven_dataloader: DataLoader,
    mocker: MockerFixture,
):
    batch_loss = mocker.patch(
        "gpt_2.pretrain.calc_loss_batch",
        side_effect=[torch.tensor(2.0), torch.tensor(10.0)],
    )

    loss = calc_loss_loader(
        uneven_dataloader,
        torch.nn.Identity(),
        torch.device("cpu"),
        num_batches=1,
    )

    batch_loss.assert_called_once()
    assert loss == pytest.approx(2.0)


@pytest.mark.parametrize("empty_samples", [False, True])
def test_calc_loss_loader_rejects_empty_evaluation(
    empty_samples: bool,
    mocker: MockerFixture,
):
    tokens = torch.empty((1 if empty_samples else 0, 0), dtype=torch.long)
    loader = DataLoader(TensorDataset(tokens, tokens.clone()), batch_size=1)
    batch_loss = mocker.patch("gpt_2.pretrain.calc_loss_batch")

    with pytest.raises(ValueError, match="evaluation tokens"):
        calc_loss_loader(loader, torch.nn.Identity(), torch.device("cpu"))

    batch_loss.assert_not_called()


@pytest.mark.parametrize("num_batches", [-1, 0, True, False, 1.5, 1.0, "1"])
def test_calc_loss_loader_rejects_invalid_num_batches(
    num_batches: object,
    uneven_dataloader: DataLoader,
    mocker: MockerFixture,
):
    batch_loss = mocker.patch("gpt_2.pretrain.calc_loss_batch")

    with pytest.raises(ValueError, match="num_batches"):
        calc_loss_loader(
            uneven_dataloader,
            torch.nn.Identity(),
            torch.device("cpu"),
            # Deliberately violate the annotation to verify runtime validation.
            num_batches=num_batches,  # ty: ignore[invalid-argument-type]
        )

    batch_loss.assert_not_called()


def test_evaluate_model(model: torch.nn.Module, dataloader: DataLoader):
    loss1, loss2 = evaluate_model(
        model,
        train_loader=dataloader,
        val_loader=dataloader,
        device=torch.device("cpu"),
        eval_iter=1,
    )

    assert loss1 >= 0 and loss2 >= 0


@pytest.mark.parametrize("train", [True, False])
def test_evaluate_model_restores_mode(
    model: torch.nn.Module,
    dataloader: DataLoader,
    train: bool,
    mocker: MockerFixture,
):
    mocker.patch("gpt_2.pretrain.calc_loss_loader", return_value=1.0)
    model.train(train)
    _, _ = evaluate_model(
        model,
        train_loader=dataloader,
        val_loader=dataloader,
        device=torch.device("cpu"),
        eval_iter=1,
    )

    assert model.training == train


@pytest.mark.parametrize("fail_on_validation", [False, True])
@pytest.mark.parametrize("train", [False, True])
def test_evaluate_model_restores_mode_after_error(
    model: torch.nn.Module,
    dataloader: DataLoader,
    fail_on_validation: bool,
    train: bool,
    mocker: MockerFixture,
):
    error = ValueError("evaluation failed")
    side_effect = [2.0, error] if fail_on_validation else error
    mocker.patch("gpt_2.pretrain.calc_loss_loader", side_effect=side_effect)
    with pytest.raises(ValueError):
        model.train(train)
        _, _ = evaluate_model(
            model,
            train_loader=dataloader,
            val_loader=dataloader,
            device=torch.device("cpu"),
            eval_iter=1,
        )

    assert model.training == train


def test_evaluate_model_loss_call_isin_eval_mode(
    model: torch.nn.Module,
    dataloader: DataLoader,
    mocker: MockerFixture,
):

    def fake_calc_loss_loader(
        data_loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        num_batches: int | None = None,
    ):
        assert model.training is False
        assert torch.is_grad_enabled() is False
        return 2.0

    mocker.patch(
        "gpt_2.pretrain.calc_loss_loader",
        autospec=True,
        side_effect=fake_calc_loss_loader,
    )
    evaluate_model(
        model,
        train_loader=dataloader,
        val_loader=dataloader,
        device=torch.device("cpu"),
        eval_iter=1,
    )


@pytest.mark.parametrize("correct_device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("correct_num_batches", [10, 11, 12])
def test_evaluate_model_loss_call_receive_correct_device_and_batch_limit(
    model: torch.nn.Module,
    mocker: MockerFixture,
    correct_device: torch.device,
    correct_num_batches: int,
):
    train_loader = mocker.Mock(spec=DataLoader)
    val_loader = mocker.Mock(spec=DataLoader)

    def fake_calc_loss_loader(
        data_loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        num_batches: int | None = None,
    ) -> float:
        assert device == correct_device
        assert num_batches == correct_num_batches
        return 1.0

    mock_loss = mocker.patch(
        "gpt_2.pretrain.calc_loss_loader",
        autospec=True,
        side_effect=fake_calc_loss_loader,
    )
    evaluate_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=correct_device,
        eval_iter=correct_num_batches,
    )

    mock_loss.assert_has_calls(
        [
            mocker.call(
                train_loader, model, correct_device, num_batches=correct_num_batches
            ),
            mocker.call(
                val_loader, model, correct_device, num_batches=correct_num_batches
            ),
        ],
        any_order=False,
    )
    assert mock_loss.call_count == 2


def test_evaluate_model_loss_returns_correct_losses(
    model: torch.nn.Module,
    dataloader: DataLoader,
    mocker: MockerFixture,
):

    mocker.patch(
        "gpt_2.pretrain.calc_loss_loader",
        autospec=True,
        side_effect=[2.0, 3.0],
    )
    train_loss, val_loss = evaluate_model(
        model,
        train_loader=dataloader,
        val_loader=dataloader,
        device=torch.device("cpu"),
        eval_iter=1,
    )

    assert train_loss == 2.0
    assert val_loss == 3.0


@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters())


def test_train_model_simple(model, optimizer, dataloader):

    loss1, loss2, toks = train_model_simple(
        model,
        train_loader=dataloader,
        val_loader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        num_epochs=1,
        eval_freq=1,
        eval_iter=1,
        start_context="Hello",
        tokenizer=tiktoken.get_encoding("gpt2"),
    )

    # Simple sanity check
    for el in loss1:
        assert el >= 0
    for el in loss2:
        assert el >= 0
    assert len(toks) >= 0


@pytest.fixture
def training_dataset() -> TensorDataset:
    # Nine four-token windows produce batches of 4, 4, and 1 samples.
    tokens = torch.arange(9 * 5, dtype=torch.long).reshape(9, 5)
    return TensorDataset(tokens[:, :-1], tokens[:, 1:])


@pytest.fixture
def train_loader(training_dataset: TensorDataset) -> DataLoader:
    return DataLoader(
        training_dataset,
        batch_size=TINY_TRAINING_CONFIG.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(TINY_TRAINING_CONFIG.seed),
    )


@pytest.fixture
def train_eval_loader(training_dataset: TensorDataset) -> DataLoader:
    # Evaluate the same training samples without sharing the shuffle generator.
    return DataLoader(
        training_dataset,
        batch_size=TINY_TRAINING_CONFIG.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(TINY_TRAINING_CONFIG.seed),
    )


@pytest.fixture
def val_loader() -> DataLoader:
    tokens = torch.arange(64, 64 + 5 * 5, dtype=torch.long).reshape(5, 5)
    return DataLoader(
        TensorDataset(tokens[:, :-1], tokens[:, 1:]),
        batch_size=TINY_TRAINING_CONFIG.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(TINY_TRAINING_CONFIG.seed),
    )


def test_train_model_perform_correct_num_updates(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    mocker: MockerFixture,
):
    initial_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    model.eval()
    step_spy = mocker.spy(optimizer, "step")
    result = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        config=TINY_TRAINING_CONFIG,
    )
    expected_steps = len(train_loader) * TINY_TRAINING_CONFIG.num_epochs
    assert step_spy.call_count == expected_steps
    assert result.global_step == expected_steps
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, model.parameters(), strict=True)
    )


def test_train_model_partial_batches_contribute_token_counts(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_dataset: TensorDataset,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
):
    result = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        config=TINY_TRAINING_CONFIG,
    )
    targets = training_dataset.tensors[1]
    assert result.tokens_seen == targets.numel() * TINY_TRAINING_CONFIG.num_epochs


@pytest.mark.parametrize("eval_every_steps", [2, 4, 10])
def test_train_model_evaluation_occurs_at_expected_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    eval_every_steps: int,
    mocker: MockerFixture,
):
    config = replace(TINY_TRAINING_CONFIG, eval_every_steps=eval_every_steps)
    mock_evaluate = mocker.patch(
        "gpt_2.pretrain.evaluate_model",
        autospec=True,
        return_value=(1.0, 2.0),
    )
    result = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        config=config,
    )

    evaluation_steps = [evaluation.global_step for evaluation in result.evaluations]
    total_steps = len(train_loader) * config.num_epochs
    evaluate_period = config.eval_every_steps

    expected_steps = list(range(evaluate_period, total_steps + 1, evaluate_period))
    if total_steps % evaluate_period != 0:
        expected_steps.append(total_steps)

    assert evaluation_steps == expected_steps
    assert mock_evaluate.call_count == len(expected_steps)


def test_train_model_receives_separate_training_evaluation_loaders(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    mocker: MockerFixture,
):
    expected_model = model
    expected_val_loader = val_loader
    device = torch.device("cpu")

    def fake_evaluate_model(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        eval_iter: int,
    ) -> tuple[float, float]:
        assert model is expected_model
        assert train_loader is train_eval_loader
        assert val_loader is expected_val_loader
        assert device == torch.device("cpu")
        assert eval_iter == TINY_TRAINING_CONFIG.eval_batches
        return 1.0, 2.0

    mock_evaluate = mocker.patch(
        "gpt_2.pretrain.evaluate_model",
        autospec=True,
        side_effect=fake_evaluate_model,
    )

    result = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        device=device,
        config=TINY_TRAINING_CONFIG,
    )

    assert result.evaluations
    assert mock_evaluate.call_count == len(result.evaluations)


def test_train_model_returns_progress_and_evaluation_records(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    mocker: MockerFixture,
):
    config = replace(TINY_TRAINING_CONFIG, eval_every_steps=4)
    mocker.patch(
        "gpt_2.pretrain.evaluate_model",
        autospec=True,
        side_effect=[(1.25, 2.5), (0.75, 1.5)],
    )

    result = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        config=config,
    )

    # Epoch one consumes 36 tokens; epoch two's first batch adds 16.
    assert result == TrainingResult(
        completed_epochs=2,
        global_step=6,
        tokens_seen=72,
        evaluations=(
            EvaluationRecord(
                epoch=2,
                global_step=4,
                tokens_seen=52,
                train_loss=1.25,
                validation_loss=2.5,
            ),
            EvaluationRecord(
                epoch=2,
                global_step=6,
                tokens_seen=72,
                train_loss=0.75,
                validation_loss=1.5,
            ),
        ),
    )


def test_train_model_rejects_empty_training_before_updates(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    mocker: MockerFixture,
):
    empty_tokens = torch.empty((0, 4), dtype=torch.long)
    empty_loader = DataLoader(TensorDataset(empty_tokens, empty_tokens.clone()))
    step_spy = mocker.spy(optimizer, "step")
    zero_grad_spy = mocker.spy(optimizer, "zero_grad")
    batch_loss = mocker.patch("gpt_2.pretrain.calc_loss_batch", autospec=True)
    evaluation = mocker.patch("gpt_2.pretrain.evaluate_model", autospec=True)

    with pytest.raises(ValueError, match="zero batches"):
        train_model(
            model=model,
            optimizer=optimizer,
            train_loader=empty_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            config=TINY_TRAINING_CONFIG,
        )

    zero_grad_spy.assert_not_called()
    batch_loss.assert_not_called()
    step_spy.assert_not_called()
    evaluation.assert_not_called()


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([5.0, 3.0, 0.0, 3.0]),
    ],
)
def test_softmax_with_temperature(logits):
    temp = 1.0
    probs_unscaled = torch.softmax(logits, dim=0)
    result = softmax_with_temperature(logits, temp)
    assert torch.all(torch.eq(probs_unscaled, result))


def test_generate(model):
    idx = torch.tensor([[56, 75, 45, 66, 77, 34]])
    result = generate(
        model,
        idx=idx,
        max_new_tokens=20,
        context_size=256,
    )

    result = result.flatten()
    idx = idx.flatten()
    assert len(result) > len(idx)
    assert torch.all(torch.eq(result[:-20], idx))


@pytest.mark.parametrize(
    "left, right",
    [
        (torch.zeros(2, 6, dtype=torch.float64), np.ones((2, 6))),
        (
            torch.zeros(3, 12, dtype=torch.float32),
            torch.ones(3, 12, dtype=torch.float64),
        ),
    ],
)
def test_assign(left, right):
    expected = torch.as_tensor(
        right,
        dtype=left.dtype,
        device=left.device,
    ).clone()
    result = assign(left, right)

    if isinstance(right, np.ndarray):
        right.fill(0)
    else:
        right.zero_()

    assert torch.equal(result.detach(), expected)
    assert result.requires_grad is True
    assert result.dtype == left.dtype
    assert result.device == left.device


def test_assign_shape_mismatch():
    with pytest.raises(ValueError):
        assign(torch.zeros(1, 2), torch.zeros(2, 2))

    with pytest.raises(ValueError):
        assign(torch.zeros(3, 2), torch.ones(3, 3))
