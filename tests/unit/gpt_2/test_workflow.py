from pathlib import Path

import pytest
import torch
from pytest_mock import MockerFixture
from torch.utils.data import DataLoader

from gpt_2.cli import PretrainArguments
from gpt_2.config import RunConfig
from gpt_2.dataset import DataLoaderBundle
from gpt_2.model import GPTModel
from gpt_2.pretrain import EvaluationRecord, TrainingResult
from gpt_2.runtime import ResolvedRuntime
from gpt_2.workflow import (
    PreparedPretrainRun,
    PretrainComponents,
    create_pretrain_components,
    create_pretrain_loaders,
    prepare_pretrain_run,
    run_pretraining,
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

    def check_output_directory(
        path: Path, *, config: RunConfig, runtime: ResolvedRuntime
    ) -> None:
        assert path.parent == outputs_dir
        assert path.parent.is_dir()

    mock_write_run_metadata = mocker.patch(
        "gpt_2.workflow.write_run_metadata",
        autospec=True,
        side_effect=check_output_directory,
    )

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )

    result = prepare_pretrain_run(arguments=arguments)
    runtime = ResolvedRuntime(
        device=torch.device("cuda"),
        seed=run_config.training.seed,
    )

    assert result == PreparedPretrainRun(
        arguments=arguments,
        config=run_config,
        runtime=runtime,
    )

    assert outputs_dir.is_dir()
    mock_load_run_config.assert_called_once_with(config_file)
    mock_resolve_device.assert_called_once_with("auto")
    mock_seed_random_sources.assert_called_once_with(run_config.training.seed)
    mock_write_run_metadata.assert_called_once_with(
        outputs_dir / "run-config.json", config=run_config, runtime=runtime
    )


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

    mock_write_run_metadata = mocker.patch(
        "gpt_2.workflow.write_run_metadata", autospec=True
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
    mock_write_run_metadata.assert_not_called()
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

    mock_write_run_metadata = mocker.patch(
        "gpt_2.workflow.write_run_metadata", autospec=True
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
    mock_write_run_metadata.assert_not_called()
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

    mock_write_run_metadata = mocker.patch(
        "gpt_2.workflow.write_run_metadata", autospec=True
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
    mock_write_run_metadata.assert_not_called()


def test_prepare_pretrain_run_write_metadata_failure(
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

    mock_write_run_metadata = mocker.patch(
        "gpt_2.workflow.write_run_metadata",
        autospec=True,
        side_effect=OSError("metadata write failed"),
    )

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )
    with pytest.raises(OSError, match="metadata"):
        prepare_pretrain_run(arguments=arguments)

    runtime = ResolvedRuntime(
        device=torch.device("cuda"),
        seed=run_config.training.seed,
    )

    assert outputs_dir.is_dir()
    mock_load_run_config.assert_called_once_with(config_file)
    mock_resolve_device.assert_called_once_with("auto")
    mock_seed_random_sources.assert_called_once_with(run_config.training.seed)
    mock_write_run_metadata.assert_called_once_with(
        outputs_dir / "run-config.json", config=run_config, runtime=runtime
    )


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

    mock_write_run_metadata = mocker.patch(
        "gpt_2.workflow.write_run_metadata", autospec=True
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
    mock_write_run_metadata.assert_not_called()
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

    mock_write_run_metadata = mocker.patch(
        "gpt_2.workflow.write_run_metadata", autospec=True
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
    mock_write_run_metadata.assert_not_called()
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


@pytest.fixture
def expected_results() -> TrainingResult:
    return TrainingResult(
        completed_epochs=2,
        global_step=100,
        tokens_seen=100_000,
        evaluations=(
            EvaluationRecord(
                epoch=1,
                global_step=50,
                tokens_seen=50_000,
                train_loss=1e-1,
                validation_loss=1e-1,
            ),
            EvaluationRecord(
                epoch=2,
                global_step=100,
                tokens_seen=100_000,
                train_loss=1e-1,
                validation_loss=1e-1,
            ),
        ),
    )


@pytest.fixture
def components(model: GPTModel) -> PretrainComponents:
    return PretrainComponents(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
    )


@pytest.fixture
def loaders(mocker: MockerFixture) -> DataLoaderBundle:
    return DataLoaderBundle(
        train=mocker.Mock(spec=DataLoader),
        train_eval=mocker.Mock(spec=DataLoader),
        validation=mocker.Mock(spec=DataLoader),
        test=mocker.Mock(spec=DataLoader),
    )


@pytest.fixture
def pt_arguments(tmp_path: Path) -> PretrainArguments:
    config_path = tmp_path / "config.toml"
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output"
    return PretrainArguments(
        config=config_path,
        input=input_path,
        output_dir=output_path,
    )


def test_run_pretraining_correctly_assembles(
    mocker: MockerFixture,
    pt_arguments: PretrainArguments,
    prepared_run: PreparedPretrainRun,
    components: PretrainComponents,
    loaders: DataLoaderBundle,
    expected_results: TrainingResult,
):
    mock_prepare = mocker.patch(
        "gpt_2.workflow.prepare_pretrain_run", autospec=True, return_value=prepared_run
    )
    mock_loaders = mocker.patch(
        "gpt_2.workflow.create_pretrain_loaders", autospec=True, return_value=loaders
    )
    mock_create_components = mocker.patch(
        "gpt_2.workflow.create_pretrain_components",
        autospec=True,
        return_value=components,
    )
    mock_train = mocker.patch(
        "gpt_2.workflow.train_model", autospec=True, return_value=expected_results
    )

    calls = mocker.Mock()
    calls.attach_mock(mock_prepare, "prepare")
    calls.attach_mock(mock_loaders, "loaders")
    calls.attach_mock(mock_create_components, "components")
    calls.attach_mock(mock_train, "train")

    result = run_pretraining(arguments=pt_arguments)

    assert [call[0] for call in calls.mock_calls] == [
        "prepare",
        "loaders",
        "components",
        "train",
    ]

    assert result is expected_results
    mock_prepare.assert_called_once_with(pt_arguments)
    mock_loaders.assert_called_once_with(prepared_run)
    mock_create_components.assert_called_once_with(prepared_run)
    mock_train.assert_called_once_with(
        model=components.model,
        optimizer=components.optimizer,
        train_loader=loaders.train,
        train_eval_loader=loaders.train_eval,
        val_loader=loaders.validation,
        device=prepared_run.runtime.device,
        config=prepared_run.config.training,
    )


def test_run_pretraining_preparation_failure(
    mocker: MockerFixture,
    pt_arguments: PretrainArguments,
    components: PretrainComponents,
    loaders: DataLoaderBundle,
    expected_results: TrainingResult,
):
    mock_prepare = mocker.patch(
        "gpt_2.workflow.prepare_pretrain_run",
        autospec=True,
        side_effect=FileNotFoundError(),
    )
    mock_loaders = mocker.patch(
        "gpt_2.workflow.create_pretrain_loaders", autospec=True, return_value=loaders
    )
    mock_create_components = mocker.patch(
        "gpt_2.workflow.create_pretrain_components",
        autospec=True,
        return_value=components,
    )
    mock_train = mocker.patch(
        "gpt_2.workflow.train_model", autospec=True, return_value=expected_results
    )

    calls = mocker.Mock()
    calls.attach_mock(mock_prepare, "prepare")
    calls.attach_mock(mock_loaders, "loaders")
    calls.attach_mock(mock_create_components, "components")
    calls.attach_mock(mock_train, "train")

    with pytest.raises(FileNotFoundError):
        run_pretraining(arguments=pt_arguments)

    assert [call[0] for call in calls.mock_calls] == [
        "prepare",
    ]

    mock_prepare.assert_called_once_with(pt_arguments)
    mock_loaders.assert_not_called()
    mock_create_components.assert_not_called()
    mock_train.assert_not_called()


def test_run_pretraining_loaders_failure(
    mocker: MockerFixture,
    pt_arguments: PretrainArguments,
    prepared_run: PreparedPretrainRun,
    components: PretrainComponents,
    expected_results: TrainingResult,
):
    mock_prepare = mocker.patch(
        "gpt_2.workflow.prepare_pretrain_run",
        autospec=True,
        return_value=prepared_run,
    )
    mock_loaders = mocker.patch(
        "gpt_2.workflow.create_pretrain_loaders",
        autospec=True,
        side_effect=ValueError(),
    )
    mock_create_components = mocker.patch(
        "gpt_2.workflow.create_pretrain_components",
        autospec=True,
        return_value=components,
    )
    mock_train = mocker.patch(
        "gpt_2.workflow.train_model", autospec=True, return_value=expected_results
    )

    calls = mocker.Mock()
    calls.attach_mock(mock_prepare, "prepare")
    calls.attach_mock(mock_loaders, "loaders")
    calls.attach_mock(mock_create_components, "components")
    calls.attach_mock(mock_train, "train")

    with pytest.raises(ValueError):
        run_pretraining(arguments=pt_arguments)

    assert [call[0] for call in calls.mock_calls] == [
        "prepare",
        "loaders",
    ]

    mock_prepare.assert_called_once_with(pt_arguments)
    mock_loaders.assert_called_once_with(prepared_run)
    mock_create_components.assert_not_called()
    mock_train.assert_not_called()


def test_run_pretraining_components_failure(
    mocker: MockerFixture,
    pt_arguments: PretrainArguments,
    prepared_run: PreparedPretrainRun,
    loaders: DataLoaderBundle,
    expected_results: TrainingResult,
):
    mock_prepare = mocker.patch(
        "gpt_2.workflow.prepare_pretrain_run",
        autospec=True,
        return_value=prepared_run,
    )
    mock_loaders = mocker.patch(
        "gpt_2.workflow.create_pretrain_loaders",
        autospec=True,
        return_value=loaders,
    )
    mock_create_components = mocker.patch(
        "gpt_2.workflow.create_pretrain_components",
        autospec=True,
        side_effect=ValueError(),
    )
    mock_train = mocker.patch(
        "gpt_2.workflow.train_model", autospec=True, return_value=expected_results
    )

    calls = mocker.Mock()
    calls.attach_mock(mock_prepare, "prepare")
    calls.attach_mock(mock_loaders, "loaders")
    calls.attach_mock(mock_create_components, "components")
    calls.attach_mock(mock_train, "train")

    with pytest.raises(ValueError):
        run_pretraining(arguments=pt_arguments)

    assert [call[0] for call in calls.mock_calls] == [
        "prepare",
        "loaders",
        "components",
    ]

    mock_prepare.assert_called_once_with(pt_arguments)
    mock_loaders.assert_called_once_with(prepared_run)
    mock_create_components.assert_called_once_with(prepared_run)
    mock_train.assert_not_called()


def test_run_pretraining_training_failure(
    mocker: MockerFixture,
    pt_arguments: PretrainArguments,
    prepared_run: PreparedPretrainRun,
    loaders: DataLoaderBundle,
    components: PretrainComponents,
):
    mock_prepare = mocker.patch(
        "gpt_2.workflow.prepare_pretrain_run",
        autospec=True,
        return_value=prepared_run,
    )
    mock_loaders = mocker.patch(
        "gpt_2.workflow.create_pretrain_loaders",
        autospec=True,
        return_value=loaders,
    )
    mock_create_components = mocker.patch(
        "gpt_2.workflow.create_pretrain_components",
        autospec=True,
        return_value=components,
    )
    mock_train = mocker.patch(
        "gpt_2.workflow.train_model",
        autospec=True,
        side_effect=ValueError(),
    )

    calls = mocker.Mock()
    calls.attach_mock(mock_prepare, "prepare")
    calls.attach_mock(mock_loaders, "loaders")
    calls.attach_mock(mock_create_components, "components")
    calls.attach_mock(mock_train, "train")

    with pytest.raises(ValueError):
        run_pretraining(arguments=pt_arguments)

    assert [call[0] for call in calls.mock_calls] == [
        "prepare",
        "loaders",
        "components",
        "train",
    ]

    mock_prepare.assert_called_once_with(pt_arguments)
    mock_loaders.assert_called_once_with(prepared_run)
    mock_create_components.assert_called_once_with(prepared_run)
    mock_train.assert_called_with(
        model=components.model,
        optimizer=components.optimizer,
        train_loader=loaders.train,
        train_eval_loader=loaders.train_eval,
        val_loader=loaders.validation,
        device=prepared_run.runtime.device,
        config=prepared_run.config.training,
    )
