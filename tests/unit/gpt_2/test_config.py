import pytest
from dataclasses import FrozenInstanceError, asdict
from gpt_2.config import ModelConfig, TrainingConfig, ensure_model_config

VALID_CONFIG = {
    "vocab_size": 128,
    "context_length": 32,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

VALID_TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 3e-4,
    "num_epochs": 2,
    "eval_every_steps": 50,
    "eval_batches": 5,
    "checkpoint_every_steps": 100,
    "seed": 0,
    "device": "auto",
}

@pytest.mark.parametrize(
    'vocab_size, context_length, emb_dim, ' \
    'n_heads, n_layers, drop_rate, qkv_bias',
    [
        (100, 1000, 250, 25, 64, 0.4, True),
        (1000, 10000, 125, 5, 128, 0.3, False),
        (300, 2000, 500, 50, 32, 0.6, True),
        (500, 20000, 1000, 100, 256, 0.8, False),
        (200, 3000, 2000, 200, 512, 0.2, True),
        (200, 3000, 2000, 200, 512, 0, True),
        (200, 3000, 2000, 200, 512, 0.0, True),
        (200, 3000, 2000, 200, 512, 1.0, True),
        (200, 3000, 2000, 200, 512, 1, True),
    ]
)
def test_accepts_valid_model_config(
    vocab_size, context_length, emb_dim,
    n_heads, n_layers, drop_rate, qkv_bias):
    ModelConfig(
        vocab_size, context_length, emb_dim,
        n_heads, n_layers, drop_rate, qkv_bias
    )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("vocab_size", 0, "vocab_size"),
        ("vocab_size", -1, "vocab_size"),
        ("vocab_size", float('inf'), "vocab_size"),
        ("vocab_size", float('-inf'), "vocab_size"),
        ("vocab_size", 2.0, "vocab_size"),
        ("vocab_size", True, "vocab_size"),
        ("context_length", -1, "context_length"),
        ("context_length", 0, "context_length"),
        ("context_length", float('inf'), "context_length"),
        ("context_length", float('-inf'), "context_length"),
        ("context_length", -1.0, "context_length"),
        ("context_length", True, "context_length"),
        ("emb_dim", -2, "emb_dim"),
        ("emb_dim", 0, "emb_dim"),
        ("emb_dim", float('inf'), "emb_dim"),
        ("emb_dim", float('-inf'), "emb_dim"),
        ("emb_dim", -3.0, "emb_dim"),
        ("emb_dim", True, "emb_dim"),
        ("emb_dim", 18, "divisible"),
        ("n_heads", -4, "n_heads"),
        ("n_heads", float('inf'), "n_heads"),
        ("n_heads", float('-inf'), "n_heads"),
        ("n_heads", 0, "n_heads"),
        ("n_heads", 4.0, "n_heads"),
        ("n_heads", True, "n_heads"),
        ("n_layers", -3, "n_layers"),
        ("n_layers", float('inf'), "n_layers"),
        ("n_layers", float('-inf'), "n_layers"),
        ("n_layers", 0, "n_layers"),
        ("n_layers", -3.0, "n_layers"),
        ("n_layers", True, "n_layers"),
        ("drop_rate", -3, "drop_rate"),
        ("drop_rate", 2, "drop_rate"),
        ("drop_rate", float('inf'), "drop_rate"),
        ("drop_rate", float('-inf'), "drop_rate"),
        ("drop_rate", float("nan"), "drop_rate"),
        ("drop_rate", -3.0, "drop_rate"),
        ("drop_rate", False, "drop_rate"),
        ("qkv_bias", -3, "qkv_bias"),
        ("qkv_bias", 0, "qkv_bias"),
        ("qkv_bias", float('inf'), "qkv_bias"),
        ("qkv_bias", float('-inf'), "qkv_bias"),
        ("qkv_bias", 2.0, "qkv_bias"),
    ],
)
def test_rejects_invalid_model_config_field(field, value, message):
    # Dictionary union creates a new dictionary, leaving the original unchanged.
    values = VALID_CONFIG | {field: value}

    with pytest.raises(ValueError, match=message):
        ModelConfig(**values)

@pytest.fixture
def model_config():
    return ModelConfig(
        vocab_size=100, context_length=100, emb_dim=120,
        n_heads=12, n_layers=20, drop_rate=0.5, qkv_bias=True
    )
def test_model_config_immutability(model_config):
    with pytest.raises(FrozenInstanceError):
        model_config.emb_dim = 'foo'

def test_model_config_to_dict(model_config):
    config_dict = asdict(model_config)
    assert isinstance(config_dict, dict)
    assert config_dict == dict(
        vocab_size=100, context_length=100, emb_dim=120,
        n_heads=12, n_layers=20, drop_rate=0.5, qkv_bias=True
    )

def test_ensure_model_config_returns_existing_instance(model_config):
    assert ensure_model_config(model_config) is model_config


def test_ensure_model_config_converts_mapping():
    result = ensure_model_config(VALID_CONFIG)

    assert result == ModelConfig(**VALID_CONFIG)

def test_ensure_model_config_validates_mapping():
    values = VALID_CONFIG | {"emb_dim": 18}

    with pytest.raises(ValueError, match="divisible"):
        ensure_model_config(values)


#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#
# TRAINING CONFIG TESTS
#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#

@pytest.mark.parametrize(
    'batch_size, learning_rate, num_epochs, ' \
    'eval_every_steps, eval_batches, checkpoint_every_steps, '\
    'seed, device',
    [
        (10, 2e-4, 5, 2, 5, 51, 67, 'auto'),
        (20, 3e-4, 7, 10, 10, 52, 0, 'mps'),
        (30, 4e-4, 20, 20, 15, 53, 1, 'cuda'),
        (40, 5e-4, 10, 15, 7, 54, 2, 'cpu'),
        (50, 3e-4, 8, 30, 8, 55, 3, 'auto'),
        (60, 2e-4, 30, 20, 9, 56, 4, 'mps'),
        (70, 3e-4, 50, 10, 2, 57, 7, 'cuda'),
        (80, 7e-4, 2, 8, 3, 58, 4, 'cpu'),
        (90, 18e-4, 1, 7, 10, 59, 5, 'cuda'),
        (100, 9e-4, 90, 25, 5, 60, 123, 'auto'),
    ]
)
def test_accepts_valid_training_config(
    batch_size, learning_rate, num_epochs,
    eval_every_steps, eval_batches, checkpoint_every_steps,
    seed, device):
    TrainingConfig(
        batch_size, learning_rate, num_epochs,
        eval_every_steps, eval_batches, checkpoint_every_steps,
        seed, device
    )

@pytest.mark.parametrize(
    "field,value,message",
    [
        ("batch_size", 0, "batch_size"),
        ("batch_size", -1, "batch_size"),
        ("batch_size", float('inf'), "batch_size"),
        ("batch_size", float('-inf'), "batch_size"),
        ("batch_size", 2.0, "batch_size"),
        ("batch_size", True, "batch_size"),
        ("num_epochs", -1, "num_epochs"),
        ("num_epochs", 0, "num_epochs"),
        ("num_epochs", float('inf'), "num_epochs"),
        ("num_epochs", float('-inf'), "num_epochs"),
        ("num_epochs", -1.0, "num_epochs"),
        ("num_epochs", True, "num_epochs"),
        ("seed", -1, "seed"),
        ("seed", float('inf'), "seed"),
        ("seed", float('-inf'), "seed"),
        ("seed", -1.0, "seed"),
        ("seed", True, "seed"),
        ("eval_every_steps", -2, "eval_every_steps"),
        ("eval_every_steps", 0, "eval_every_steps"),
        ("eval_every_steps", float('inf'), "eval_every_steps"),
        ("eval_every_steps", float('-inf'), "eval_every_steps"),
        ("eval_every_steps", -3.0, "eval_every_steps"),
        ("eval_every_steps", True, "eval_every_steps"),
        ("eval_batches", -4, "eval_batches"),
        ("eval_batches", float('inf'), "eval_batches"),
        ("eval_batches", float('-inf'), "eval_batches"),
        ("eval_batches", 0, "eval_batches"),
        ("eval_batches", 4.0, "eval_batches"),
        ("eval_batches", True, "eval_batches"),
        ("checkpoint_every_steps", -3, "checkpoint_every_steps"),
        ("checkpoint_every_steps", float('inf'), "checkpoint_every_steps"),
        ("checkpoint_every_steps", float('-inf'), "checkpoint_every_steps"),
        ("checkpoint_every_steps", 0, "checkpoint_every_steps"),
        ("checkpoint_every_steps", -3.0, "checkpoint_every_steps"),
        ("checkpoint_every_steps", True, "checkpoint_every_steps"),
        ("learning_rate", -3, "learning_rate"),
        ("learning_rate", float('inf'), "learning_rate"),
        ("learning_rate", float('-inf'), "learning_rate"),
        ("learning_rate", float("nan"), "learning_rate"),
        ("learning_rate", -3.0, "learning_rate"),
        ("learning_rate", False, "learning_rate"),
        ("learning_rate", 0.0, "learning_rate"),
        ("device", -3, "device"),
        ("device", 0, "device"),
        ("device", float('inf'), "device"),
        ("device", float('-inf'), "device"),
        ("device", 2.0, "device"),
        ("device", 'foo', "device"),
        ("device", None, "device"),
    ],
)
def test_rejects_invalid_training_config_field(field, value, message):
    # Dictionary union creates a new dictionary, leaving the original unchanged.
    values = VALID_TRAINING_CONFIG | {field: value}

    with pytest.raises(ValueError, match=message):
        TrainingConfig(**values)

@pytest.fixture
def training_config():
    return TrainingConfig(
        **VALID_TRAINING_CONFIG
    )
def test_immutability(training_config):
    with pytest.raises(FrozenInstanceError):
        training_config.num_epochs = 'foo'

def test_to_dict(training_config):
    config_dict = asdict(training_config)
    assert isinstance(config_dict, dict)
    assert config_dict == dict(
        **VALID_TRAINING_CONFIG
    )