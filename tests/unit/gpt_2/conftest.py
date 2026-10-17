import pytest

from gpt_2.config import ModelConfig
from gpt_2.model import GPTModel

TINY_GPT_CONFIG = ModelConfig(
    vocab_size=50257,
    context_length=32,
    emb_dim=16,
    n_heads=4,
    n_layers=1,
    drop_rate=0.0,
    qkv_bias=False,
)


@pytest.fixture
def model():
    return GPTModel(TINY_GPT_CONFIG)
