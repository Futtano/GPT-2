from pathlib import Path

import pytest
import torch
from pytest_mock import MockerFixture

from gpt_2.cli import PretrainArguments
from gpt_2.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from gpt_2.dataset import DataLoaderBundle
from gpt_2.model import GPTModel
from gpt_2.runtime import ResolvedRuntime
from gpt_2.workflow import (
    PreparedPretrainRun,
    create_pretrain_components,
    create_pretrain_loaders,
    prepare_pretrain_run,
)


@pytest.fixture
def run_config() -> RunConfig:
    return RunConfig(
        model=ModelConfig(
            vocab_size=128,
            context_length=32,
            emb_dim=16,
            n_heads=4,
            n_layers=1,
            drop_rate=0.0,
            qkv_bias=False,
        ),
        data=DataConfig(
            train_fraction=0.7,
            validation_fraction=0.15,
            stride=100,
            num_workers=2,
        ),
        training=TrainingConfig(
            batch_size=8,
            learning_rate=3e-4,
            num_epochs=2,
            eval_every_steps=50,
            eval_batches=5,
            checkpoint_every_steps=100,
            seed=0,
            device="auto",
        ),
    )


@pytest.fixture
def runtime(run_config: RunConfig) -> ResolvedRuntime:
    return ResolvedRuntime(
        device=torch.device("cpu"),
        seed=run_config.training.seed,
    )


@pytest.fixture
def prepared_run(
    tmp_path: Path,
    run_config: RunConfig,
    runtime: ResolvedRuntime,
) -> PreparedPretrainRun:
    return PreparedPretrainRun(
        arguments=PretrainArguments(
            config=tmp_path / "pretrain-tiny.toml",
            input=tmp_path / "training.txt",
            output_dir=tmp_path / "run-001",
        ),
        config=run_config,
        runtime=runtime,
    )


def test_prepare_pretrain_run_valid_input(
    tmp_path: Path, mocker: MockerFixture, run_config: RunConfig
):
    config_file = tmp_path / "pretrain-tiny.toml"
    input_file = tmp_path / "training.txt"
    outputs_dir = tmp_path / "run-001"

    config_file.touch()
    input_file.write_text("Some input text", encoding="utf-8")

    mock_load_run_config = mocker.patch(
        "gpt_2.workflow.load_run_config",
        autospec=True,
        return_value=run_config,
    )

    mock_resolve_device = mocker.patch(
        "gpt_2.workflow.resolve_device",
        autospec=True,
        return_value=torch.device("cuda"),
    )

    mock_seed_random_sources = mocker.patch(
        "gpt_2.workflow.seed_random_sources",
        autospec=True,
    )

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )

    result = prepare_pretrain_run(arguments=arguments)

    assert result == PreparedPretrainRun(
        arguments=arguments,
        config=run_config,
        runtime=ResolvedRuntime(
            device=torch.device("cuda"),
            seed=run_config.training.seed,
        ),
    )

    assert outputs_dir.is_dir()
    mock_load_run_config.assert_called_once_with(config_file)
    mock_resolve_device.assert_called_once_with("auto")
    mock_seed_random_sources.assert_called_once_with(run_config.training.seed)


def test_prepare_pretrain_run_raises_if_config_file_not_exists(
    tmp_path: Path, mocker: MockerFixture, run_config: RunConfig
):
    config_file = tmp_path / "pretrain-tiny.toml"
    input_file = tmp_path / "training.txt"
    outputs_dir = tmp_path / "run-001"

    input_file.write_text("Some input text", encoding="utf-8")

    mock_load_run_config = mocker.patch(
        "gpt_2.workflow.load_run_config",
        autospec=True,
        return_value=run_config,
    )

    mock_resolve_device = mocker.patch(
        "gpt_2.workflow.resolve_device",
        autospec=True,
        return_value=torch.device("cuda"),
    )

    mock_seed_random_sources = mocker.patch(
        "gpt_2.workflow.seed_random_sources",
        autospec=True,
    )

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )

    with pytest.raises(FileNotFoundError, match="Configuration"):
        prepare_pretrain_run(arguments=arguments)

    mock_load_run_config.assert_not_called()
    mock_resolve_device.assert_not_called()
    mock_seed_random_sources.assert_not_called()
    assert not outputs_dir.exists()


def test_prepare_pretrain_run_raises_if_input_file_not_exists(
    tmp_path: Path,
    mocker: MockerFixture,
    run_config: RunConfig,
):
    config_file = tmp_path / "pretrain-tiny.toml"
    input_file = tmp_path / "training.txt"
    outputs_dir = tmp_path / "run-001"

    config_file.touch()

    mock_load_run_config = mocker.patch(
        "gpt_2.workflow.load_run_config",
        autospec=True,
        return_value=run_config,
    )

    mock_resolve_device = mocker.patch(
        "gpt_2.workflow.resolve_device",
        autospec=True,
        return_value=torch.device("cuda"),
    )

    mock_seed_random_sources = mocker.patch(
        "gpt_2.workflow.seed_random_sources",
        autospec=True,
    )

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )

    with pytest.raises(FileNotFoundError, match="Input"):
        prepare_pretrain_run(arguments=arguments)

    mock_load_run_config.assert_not_called()
    mock_resolve_device.assert_not_called()
    mock_seed_random_sources.assert_not_called()
    assert not outputs_dir.exists()


def test_prepare_pretrain_run_raises_if_output_dir_exists(
    tmp_path: Path, mocker: MockerFixture, run_config: RunConfig
):
    config_file = tmp_path / "pretrain-tiny.toml"
    input_file = tmp_path / "training.txt"
    outputs_dir = tmp_path / "run-001"

    input_file.write_text("Some input text", encoding="utf-8")
    config_file.touch()
    outputs_dir.mkdir(parents=True)

    mocker.patch(
        "gpt_2.workflow.load_run_config",
        autospec=True,
        return_value=run_config,
    )

    mock_resolve_device = mocker.patch(
        "gpt_2.workflow.resolve_device",
        autospec=True,
        return_value=torch.device("cuda"),
    )

    mock_seed_random_sources = mocker.patch(
        "gpt_2.workflow.seed_random_sources",
        autospec=True,
    )

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )

    with pytest.raises(FileExistsError):
        prepare_pretrain_run(arguments=arguments)

    assert outputs_dir.exists()
    mock_resolve_device.assert_not_called()
    mock_seed_random_sources.assert_not_called()


def test_invalid_config_does_not_create_output(
    tmp_path: Path,
    mocker: MockerFixture,
):
    config_file = tmp_path / "config.toml"
    input_file = tmp_path / "training.txt"
    output_dir = tmp_path / "output"

    config_file.touch()
    input_file.write_text("Training text", encoding="utf-8")

    mock_load_run_config = mocker.patch(
        "gpt_2.workflow.load_run_config",
        autospec=True,
        side_effect=ValueError("Invalid configuration"),
    )

    mock_resolve_device = mocker.patch(
        "gpt_2.workflow.resolve_device",
        autospec=True,
        return_value=torch.device("cuda"),
    )

    mock_seed_random_sources = mocker.patch(
        "gpt_2.workflow.seed_random_sources",
        autospec=True,
    )

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="Invalid configuration"):
        prepare_pretrain_run(arguments)

    mock_load_run_config.assert_called_once_with(config_file)
    mock_resolve_device.assert_not_called()
    mock_seed_random_sources.assert_not_called()
    assert not output_dir.exists()


def test_prepare_pretrain_fails_if_device_resolution_fails(
    tmp_path: Path, run_config: RunConfig, mocker: MockerFixture
):
    config_file = tmp_path / "pretrain-tiny.toml"
    input_file = tmp_path / "training.txt"
    outputs_dir = tmp_path / "run-001"

    config_file.touch()
    input_file.write_text("Some input text", encoding="utf-8")

    mock_load_run_config = mocker.patch(
        "gpt_2.workflow.load_run_config",
        autospec=True,
        return_value=run_config,
    )

    mock_resolve_device = mocker.patch(
        "gpt_2.workflow.resolve_device",
        autospec=True,
        side_effect=RuntimeError("accelerator cuda is not available on your machine."),
    )

    mock_seed_random_sources = mocker.patch(
        "gpt_2.workflow.seed_random_sources",
        autospec=True,
    )

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )

    with pytest.raises(RuntimeError, match="cuda"):
        prepare_pretrain_run(arguments=arguments)

    mock_load_run_config.assert_called_once_with(config_file)
    mock_resolve_device.assert_called_once_with("auto")
    mock_seed_random_sources.assert_not_called()
    assert not outputs_dir.exists()


def test_create_pretrain_loaders(
    mocker: MockerFixture,
    prepared_run: PreparedPretrainRun,
):
    run = prepared_run
    token_ids = [1, 2, 3, 4]
    bundle = mocker.Mock(spec=DataLoaderBundle)

    mock_load_tokens = mocker.patch(
        "gpt_2.workflow.load_text_token_ids",
        autospec=True,
        return_value=token_ids,
    )
    mock_create_loaders = mocker.patch(
        "gpt_2.workflow.create_data_loaders",
        autospec=True,
        return_value=bundle,
    )

    result = create_pretrain_loaders(run)

    mock_load_tokens.assert_called_once_with(
        run.arguments.input,
        model_config=run.config.model,
    )
    mock_create_loaders.assert_called_once_with(
        token_ids,
        model_config=run.config.model,
        training_config=run.config.training,
        data_config=run.config.data,
    )
    assert result is bundle


def test_create_pretrain_loaders_token_loading_error_prevents_loader_construction(
    mocker: MockerFixture,
    prepared_run: PreparedPretrainRun,
):
    run = prepared_run
    mock_load_tokens = mocker.patch(
        "gpt_2.workflow.load_text_token_ids",
        autospec=True,
        side_effect=ValueError("token loading failure"),
    )
    mock_create_loaders = mocker.patch(
        "gpt_2.workflow.create_data_loaders",
        autospec=True,
    )

    with pytest.raises(ValueError, match="token loading failure"):
        create_pretrain_loaders(run)

    mock_load_tokens.assert_called_once_with(
        path=run.arguments.input, model_config=run.config.model
    )
    mock_create_loaders.assert_not_called()


def test_create_pretrain_loaders_loading_error_propagates_to_caller(
    mocker: MockerFixture,
    prepared_run: PreparedPretrainRun,
):
    run = prepared_run
    token_ids = [1, 2, 3, 4, 5]

    mock_load_tokens = mocker.patch(
        "gpt_2.workflow.load_text_token_ids",
        autospec=True,
        return_value=token_ids,
    )
    mock_create_loaders = mocker.patch(
        "gpt_2.workflow.create_data_loaders",
        autospec=True,
        side_effect=ValueError("loader failure"),
    )

    with pytest.raises(ValueError, match="loader failure"):
        create_pretrain_loaders(run)

    mock_load_tokens.assert_called_once_with(
        path=run.arguments.input, model_config=run.config.model
    )
    mock_create_loaders.assert_called_once_with(
        token_ids=token_ids,
        model_config=run.config.model,
        training_config=run.config.training,
        data_config=run.config.data,
    )


def test_create_pretrain_components_handles_valid_configs(
    mocker: MockerFixture,
    prepared_run: PreparedPretrainRun,
):
    run = prepared_run
    move_to_device = mocker.spy(GPTModel, "to")

    pretrain_components = create_pretrain_components(
        run=run,
    )

    model, optimizer = pretrain_components.model, pretrain_components.optimizer

    move_to_device.assert_called_once_with(model, run.runtime.device)
    assert model.config == run.config.model
    for group in optimizer.param_groups:
        assert group["lr"] == run.config.training.learning_rate
        assert group["weight_decay"] == 0.0

    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    model_parameters = list(model.parameters())

    assert len(optimizer_parameters) == len(model_parameters)
    assert {id(p) for p in optimizer_parameters} == {id(p) for p in model_parameters}

    for param in model.parameters():
        assert param.device.type == "cpu"
