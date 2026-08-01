import pytest
import torch
from gpt_2.model import (
    GPT_CONFIG_124M, LayerNorm, GELU, FeedForward,
    TransformerBlock, GPTModel
)

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
    'input, cfg',
    [
        (torch.randn(2, 10, 15), {'emb_dim': 15}),
        (torch.randn(3, 10, 20), {'emb_dim': 20}),
        (torch.randn(1, 10, 10), {'emb_dim': 10}),
    ]
)
def test_feed_forward(input, cfg):
    ff = FeedForward(cfg)
    output = ff(input)
    assert output.shape == input.shape

@pytest.mark.parametrize(
    'input, cfg',
    [
        (torch.randn(2, 10, GPT_CONFIG_124M['emb_dim']), GPT_CONFIG_124M),
        (torch.randn(3, 10, GPT_CONFIG_124M['emb_dim']), GPT_CONFIG_124M),
        (torch.randn(1, 10, GPT_CONFIG_124M['emb_dim']), GPT_CONFIG_124M),
    ]
)
def test_transformer_block(input, cfg):
    tb = TransformerBlock(cfg)
    output = tb(input)
    assert output.shape == input.shape


@pytest.mark.parametrize(
    'input, cfg',
    [
        (torch.randint(low=0, high=GPT_CONFIG_124M['vocab_size'], size=(2, 10)), GPT_CONFIG_124M),
        (torch.randint(low=0, high=GPT_CONFIG_124M['vocab_size'], size=(3, 10)), GPT_CONFIG_124M),
        (torch.randint(low=0, high=GPT_CONFIG_124M['vocab_size'], size=(1, 10)), GPT_CONFIG_124M),
    ]
)
def test_gpt_model(input, cfg):
    gpt = GPTModel(cfg)
    output = gpt(input)
    assert output.shape == (input.shape[0], input.shape[1], cfg['vocab_size'])