import json
import math
import tomllib
from collections.abc import Generator
from pathlib import Path

import pytest

from gpt_2.workflow import PretrainArguments, run_pretraining

TOML_CONFIG = """
[model]
vocab_size = 50257
context_length = 8
emb_dim = 8
n_heads = 2
n_layers = 1
drop_rate = 0.0
qkv_bias = false

[training]
batch_size = 2
learning_rate = 0.001
num_epochs = 1
eval_every_steps = 2
eval_batches = 1
checkpoint_every_steps = 2
seed = 123
device = "cpu"

[data]
train_fraction = 0.6
validation_fraction = 0.2
stride = 2
num_workers = 0
"""


@pytest.fixture
def input_file(tmp_path: Path) -> Generator:
    input_file = tmp_path / "input.txt"
    input_file.write_text("Hello world!" * 40, encoding="utf-8")
    yield input_file
    input_file.unlink(missing_ok=True)


@pytest.fixture
def toml_config(tmp_path: Path) -> Generator:
    config_file = tmp_path / "config.toml"
    config_file.write_text(TOML_CONFIG, encoding="utf-8")
    yield config_file
    config_file.unlink(missing_ok=True)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "output"
    return output_dir


@pytest.fixture
def arguments(
    input_file: Path,
    toml_config: Path,
    output_dir: Path,
) -> PretrainArguments:
    return PretrainArguments(
        input=input_file,
        config=toml_config,
        output_dir=output_dir,
    )


def test_integration_run_pretraining(output_dir: Path, arguments: PretrainArguments):
    result = run_pretraining(arguments)
    assert output_dir.is_dir()
    assert result.completed_epochs == 1
    assert result.global_step > 0
    assert result.tokens_seen > 0
    assert result.evaluations
    final = result.evaluations[-1]
    assert final.epoch == result.completed_epochs
    assert final.global_step == result.global_step
    assert final.tokens_seen == result.tokens_seen
    for record in result.evaluations:
        assert math.isfinite(record.train_loss)
        assert math.isfinite(record.validation_loss)

    metadata_path = output_dir / "run-config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "schema_version": 1,
        "config": tomllib.loads(TOML_CONFIG),
        "runtime": {"device": "cpu", "seed": 123},
    }
