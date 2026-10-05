from dataclasses import dataclass

import torch

from gpt_2.cli import PretrainArguments
from gpt_2.config import RunConfig, load_run_config
from gpt_2.dataset import DataLoaderBundle, create_data_loaders, load_text_token_ids
from gpt_2.model import GPTModel
from gpt_2.runtime import ResolvedRuntime, resolve_device, seed_random_sources


@dataclass(frozen=True)
class PreparedPretrainRun:
    arguments: PretrainArguments
    config: RunConfig
    runtime: ResolvedRuntime


@dataclass(frozen=True)
class PretrainComponents:
    model: GPTModel
    optimizer: torch.optim.AdamW


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

    if arguments.output_dir.exists():
        raise FileExistsError(f"{arguments.output_dir} already exists.")

    resolved_device = resolve_device(name=config.training.device)
    seed_random_sources(seed=config.training.seed)
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = ResolvedRuntime(device=resolved_device, seed=config.training.seed)

    return PreparedPretrainRun(
        arguments=arguments,
        config=config,
        runtime=runtime,
    )


def create_pretrain_loaders(
    run: PreparedPretrainRun,
) -> DataLoaderBundle:
    token_ids = load_text_token_ids(
        path=run.arguments.input,
        model_config=run.config.model,
    )

    bundle = create_data_loaders(
        token_ids=token_ids,
        model_config=run.config.model,
        training_config=run.config.training,
        data_config=run.config.data,
    )

    return bundle


def create_pretrain_components(
    run: PreparedPretrainRun,
) -> PretrainComponents:
    model = GPTModel(
        cfg=run.config.model,
    )

    model.to(run.runtime.device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=run.config.training.learning_rate,
        weight_decay=0.0,
    )

    return PretrainComponents(
        model=model,
        optimizer=optimizer,
    )
