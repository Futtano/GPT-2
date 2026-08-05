import torch
import tiktoken
import pytest
from gpt_2.dataset import create_data_loader_v1
from gpt_2.model import GPTModel, GPT_CONFIG_124M
from gpt_2.pretrain import (
    text_to_token_ids, token_ids_to_text,
    calc_loss_batch, calc_loss_loader,
    evaluate_model, train_model_simple,
    softmax_with_temperature, generate,
    assign
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
    assert loss.item() >= 0 

@pytest.fixture
def dataloader():
    txt = "Hello from Daniele!"
    return create_data_loader_v1(
        txt,
        batch_size=2,
        max_length=4,
        stride = 1,
        shuffle=False,
        num_workers=0,
    )

def test_calc_loss_loader(dataloader, model):
    loss = calc_loss_loader(
        dataloader,
        model,
        torch.device('cpu'),
        )
    assert loss >= 0

def test_evaluate_model(model, dataloader):
    loss1, loss2 = evaluate_model(
        model,
        train_loader=dataloader,
        val_loader=dataloader,
        device=torch.device('cpu'),
        eval_iter=1)

    assert loss1 >= 0 and loss2 >= 0

@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters())

def test_train_model_simple(model, optimizer, dataloader):

    loss1, loss2, toks = train_model_simple(
        model, train_loader=dataloader, val_loader=dataloader,
        optimizer=optimizer, device=torch.device('cpu'),
        num_epochs=1, eval_freq=1, eval_iter=1, start_context='Hello',
        tokenizer=tiktoken.get_encoding('gpt2')
    )

    # Simple sanity check
    for el in loss1:
        assert el >= 0
    for el in loss2:
        assert el >= 0
    assert len(toks) >= 0

@pytest.mark.parametrize(
    'logits',
    [
        torch.tensor([1., 2., 3., 4.]),
        torch.tensor([5., 3., 0., 3.]),
    ]
)
def test_softmax_with_temperature(logits):
    temp = 1.
    probs_unscaled = torch.softmax(logits, dim=0)
    result = softmax_with_temperature(logits, temp)
    assert torch.all(torch.eq(probs_unscaled, result))

def test_generate(model):
    idx = torch.tensor([[56, 75, 45, 66, 77, 34]])
    result = generate(
        model,
        idx=idx,
        max_new_tokens=20,
        context_size=256,
    )

    result = result.flatten()
    idx = idx.flatten()
    assert len(result) > len(idx)
    assert torch.all(torch.eq(result[:-20], idx))

def test_assign():
    with pytest.raises(ValueError):
        assign(torch.zeros(1,2), torch.zeros(2, 2))

    with pytest.raises(ValueError):
        assign(torch.zeros(3,2), torch.ones(3, 3))

    assert torch.all(torch.eq(
        assign(torch.zeros(1, 5), torch.ones(1, 5)).data,
        torch.ones(1, 5)
    ))