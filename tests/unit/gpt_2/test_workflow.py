import pytest

from gpt_2.cli import PretrainArguments
from gpt_2.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from gpt_2.workflow import PreparedPretrainRun, prepare_pretrain_run


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


def test_prepare_pretrain_run_valid_input(tmp_path, mocker, run_config):
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

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )

    result = prepare_pretrain_run(arguments=arguments)

    assert result == PreparedPretrainRun(
        arguments=arguments,
        config=run_config,
    )

    assert outputs_dir.is_dir()
    mock_load_run_config.assert_called_once_with(config_file)


def test_prepare_pretrain_run_raises_if_config_file_not_exists(
    tmp_path, mocker, run_config
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

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )

    with pytest.raises(FileNotFoundError, match="Configuration"):
        prepare_pretrain_run(arguments=arguments)

    mock_load_run_config.assert_not_called()
    assert not outputs_dir.exists()


def test_prepare_pretrain_run_raises_if_input_file_not_exists(
    tmp_path, mocker, run_config
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

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )

    with pytest.raises(FileNotFoundError, match="Input"):
        prepare_pretrain_run(arguments=arguments)

    mock_load_run_config.assert_not_called()
    assert not outputs_dir.exists()


def test_prepare_pretrain_run_raises_if_output_dir_exists(tmp_path, mocker, run_config):
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

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=outputs_dir,
    )

    with pytest.raises(FileExistsError):
        prepare_pretrain_run(arguments=arguments)

    assert outputs_dir.exists()


def test_invalid_config_does_not_create_output(
    tmp_path,
    mocker,
    run_config,
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

    arguments = PretrainArguments(
        config=config_file,
        input=input_file,
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="Invalid configuration"):
        prepare_pretrain_run(arguments)

    mock_load_run_config.assert_called_once_with(config_file)
    assert not output_dir.exists()
