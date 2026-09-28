from dataclasses import dataclass

from gpt_2.cli import PretrainArguments
from gpt_2.config import RunConfig, load_run_config


@dataclass(frozen=True)
class PreparedPretrainRun:
    arguments: PretrainArguments
    config: RunConfig


def prepare_pretrain_run(
    arguments: PretrainArguments,
) -> PreparedPretrainRun:
    if not arguments.config.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {arguments.config}"
        )

    if not arguments.input.is_file():
        raise FileNotFoundError(f"Input text file does not exist: {arguments.input}")

    config = load_run_config(arguments.config)

    arguments.output_dir.mkdir(parents=True, exist_ok=False)

    return PreparedPretrainRun(
        arguments=arguments,
        config=config,
    )
