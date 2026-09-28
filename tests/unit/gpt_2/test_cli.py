from pathlib import Path

from gpt_2.cli import PretrainArguments, parse_args


def test_parser_args_returns_typed_paths():
    result = parse_args(
        [
            "--config",
            "configs/pretrain-tiny.toml",
            "--input",
            "data/training.txt",
            "--output-dir",
            "outputs/run-001",
        ]
    )

    assert result == PretrainArguments(
        config=Path("configs/pretrain-tiny.toml"),
        input=Path("data/training.txt"),
        output_dir=Path("outputs/run-001"),
    )
