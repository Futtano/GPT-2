import pytest
import torch

from gpt_2.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from gpt_2.model import GPTModel
from gpt_2.runtime import ResolvedRuntime


@pytest.fixture
def runtime(run_config: RunConfig) -> ResolvedRuntime:
    return ResolvedRuntime(
        device=torch.device("cpu"),
        seed=run_config.training.seed,
    )


@pytest.fixture
def model():
    return GPTModel(
        ModelConfig(
            vocab_size=50257,
            context_length=32,
            emb_dim=16,
            n_heads=4,
            n_layers=1,
            drop_rate=0.0,
            qkv_bias=False,
        )
    )


@pytest.fixture
def training_config():
    return TrainingConfig(
        batch_size=4,
        learning_rate=1e-3,
        num_epochs=2,
        eval_every_steps=2,
        eval_batches=2,
        checkpoint_every_steps=2,
        seed=123,
        device="cpu",
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
