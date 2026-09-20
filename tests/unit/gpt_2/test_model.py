import pytest
import torch

from dataclasses import asdict

from gpt_2.config import ModelConfig
from gpt_2.model import (
    LayerNorm, GELU, FeedForward,
    TransformerBlock, GPTModel
)

TEST_GPT_CONFIG = ModelConfig(**{
    "vocab_size": 128,
    "context_length": 16,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
})

@pytest.mark.parametrize(
    'input',
    [
        torch.randn(2, 10, 15),
        torch.randn(3, 10, 20),
        torch.randn(1, 10, 10),
    ]
)
def test_layer_norm(input):
    n_dim = input.shape[-1]
    layer_norm = LayerNorm(n_dim)
    output = layer_norm(input)
    assert output.shape == input.shape

    torch_ln = torch.nn.LayerNorm(n_dim)

    with torch.no_grad():
        torch_ln.weight.copy_(layer_norm.scale)
        torch_ln.bias.copy_(layer_norm.shift)

    expected = torch_ln(input)
    actual = layer_norm(input)

    assert torch.allclose(actual, expected, atol=1e-3)


@pytest.mark.parametrize(
    'input',
    [
        torch.randn(2, 10, 15),
        torch.randn(3, 10, 20),
        torch.randn(1, 10, 10),
    ]
)
def test_gelu(input):
    gelu = GELU()
    output = gelu(input)
    assert output.shape == input.shape

    torch_gelu = torch.nn.GELU()
    expected = torch_gelu(input)
    actual = gelu(input)
    assert torch.allclose(actual, expected, atol=1e-3)

@pytest.mark.parametrize(
    'input',
    [
        (torch.randn(2, 10, 15)),
        (torch.randn(3, 10, 20)),
        (torch.randn(1, 10, 10)),
    ]
)
def test_feed_forward(input):
    ff = FeedForward(input.shape[-1])
    output = ff(input)
    assert output.shape == input.shape

@pytest.mark.parametrize(
    'input',
    [
        (torch.randn(2, 10, TEST_GPT_CONFIG.emb_dim)),
        (torch.randn(3, 10, TEST_GPT_CONFIG.emb_dim)),
        (torch.randn(1, 10, TEST_GPT_CONFIG.emb_dim)),
    ]
)
def test_transformer_block(input):
    tb = TransformerBlock(TEST_GPT_CONFIG)
    output = tb(input)
    assert output.shape == input.shape


@pytest.mark.parametrize(
    'input',
    [
        (torch.randint(low=0, high=TEST_GPT_CONFIG.vocab_size, size=(2, 10))),
        (torch.randint(low=0, high=TEST_GPT_CONFIG.vocab_size, size=(3, 10))),
        (torch.randint(low=0, high=TEST_GPT_CONFIG.vocab_size, size=(1, 10))),
    ]
)
def test_gpt_model(input):
    gpt = GPTModel(TEST_GPT_CONFIG)
    output = gpt(input)
    assert output.shape == (input.shape[0], input.shape[1], TEST_GPT_CONFIG.vocab_size)

def test_gpt_model_accepts_mapping():
    model = GPTModel(asdict(TEST_GPT_CONFIG))

    assert model.config == TEST_GPT_CONFIG