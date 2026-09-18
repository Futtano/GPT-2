import pytest
from dataclasses import FrozenInstanceError, asdict
from gpt_2.config import ModelConfig

VALID_CONFIG = {
    "vocab_size": 128,
    "context_length": 32,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
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
def test_accepts_valid_config(
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
def test_rejects_invalid_field(field, value, message):
    values = VALID_CONFIG | {field: value} # this updates the dict

    with pytest.raises(ValueError, match=message):
        ModelConfig(**values)

@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=100, context_length=100, emb_dim=120,
        n_heads=12, n_layers=20, drop_rate=0.5, qkv_bias=True
    )
def test_immutability(config):
    with pytest.raises(FrozenInstanceError):
        config.emb_dim = 'foo'

def test_to_dict(config):
    config_dict = asdict(config)
    assert isinstance(config_dict, dict)
    assert config_dict == dict(
        vocab_size=100, context_length=100, emb_dim=120,
        n_heads=12, n_layers=20, drop_rate=0.5, qkv_bias=True
    )
