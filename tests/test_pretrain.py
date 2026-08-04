import torch
import tiktoken
import pytest
from gpt_2.model import GPTModel, GPT_CONFIG_124M
from gpt_2.pretrain import (
    text_to_token_ids, token_ids_to_text,
    calc_loss_batch
)

@pytest.mark.parametrize(
        'input',
        [
            'My name is Daniele',
            'Hello, foo!'
            '',
        ]
)

def test_text_to_token_ids(input):
    tokenizer = tiktoken.get_encoding('gpt2')
    token_ids = text_to_token_ids(input, tokenizer)

    # Must return a tensor
    assert isinstance(token_ids, torch.Tensor)
    # Must be 2-dimensional with first dimension equal to 1
    assert token_ids.shape[0] == 1

    # if empty string, an empty list must be returned
    if len(input) == 0:
        assert len(token_ids) == 0
    else: # each item must be an integer index
        for el in token_ids[0]:
            assert isinstance(el.item(), int)

@pytest.mark.parametrize(
        'input',
        [
            torch.tensor([]).unsqueeze(0),
            torch.tensor([1, 2, 3, 4]).unsqueeze(0),
            torch.tensor([1, 4, 5, 7, 50256]).unsqueeze(0),
        ]
)

def test_token_ids_to_text(input):
    tokenizer = tiktoken.get_encoding('gpt2')
    text = token_ids_to_text(input, tokenizer)

    # Must return a string
    assert isinstance(text, str)

    # if empty tensor, an empty string must be returned
    if len(input) == 0:
        assert len(text) == 0


@pytest.fixture
def text():
    return 'Hello I am Daniele'

@pytest.fixture
def batches(text):
    context_length = 3
    stride = 1
    tokenizer = tiktoken.get_encoding('gpt2')
    token_ids = tokenizer.encode(text)
    input_batch, target_batch = [], []
    for i in range(0, len(token_ids) - context_length, stride):
        input_batch.append(torch.tensor(token_ids[i : i + context_length]))
        target_batch.append(torch.tensor(token_ids[i+1 : i + context_length + 1]))

    input_batch = torch.stack(input_batch, dim=0)    
    target_batch = torch.stack(target_batch, dim=0)

    return input_batch, target_batch

@pytest.fixture
def model():
    return GPTModel(GPT_CONFIG_124M)

def test_calc_loss_batch(batches, model):
    input_batch, target_batch = batches
    loss = calc_loss_batch(input_batch, target_batch, model, torch.device('cpu'))
    assert loss.item() > 0 