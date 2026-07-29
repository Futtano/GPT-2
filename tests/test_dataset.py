import math
from pathlib import Path
import pytest
import tiktoken
from gpt_2.dataset import (
    create_data_loader_v1,
    download_pt_dataset,
    GPTDatasetV1
)

TEXT_LENGTH = 20479

@pytest.fixture
def gpt_dataset_args(tmp_path):
    file_path = download_pt_dataset(tmp_path / 'foo.txt')
    with open(file_path, 'r', encoding='utf-8') as fp:
        txt = fp.read()

    tokenizer = tiktoken.get_encoding('gpt2')

    return {'txt': txt, 'tokenizer': tokenizer}

@pytest.fixture
def out_file(request):
    yield request.param

    if request.param is not None:
        Path(request.param).unlink(missing_ok=True)

@pytest.mark.parametrize(
    "out_file, expected",
    [
        (None, 'data/the-verdict.txt'),
        ('data/foo.txt', 'data/foo.txt')
    ],
    indirect=['out_file']
)

def test_download_pt_dataset(out_file, expected):
    # default folder 
    if out_file is None:
        result = download_pt_dataset()
    else: 
        result = download_pt_dataset(out_file)
    
    assert result == expected

    with open(result, 'r', encoding='utf-8') as fp:
        txt = fp.read()

    # Correctly downloaded the full text of The Verdict
    assert len(txt) == TEXT_LENGTH

@pytest.mark.parametrize(
    "max_length, stride",
    [
        (8, 4),
        (4, 4),
    ]
)
def test_gpt_dataset(gpt_dataset_args, max_length, stride):
    dataset = GPTDatasetV1(**gpt_dataset_args, max_length=max_length, stride=stride)
    txt, tokenizer = gpt_dataset_args['txt'], gpt_dataset_args['tokenizer']
    n_tokens = len(tokenizer.encode(txt))

    # Check if the created dataset has the correct number of windows
    n_windows = math.floor((n_tokens - max_length - 1) / stride) + 1
    n_windows = max(0, n_windows)
    assert len(dataset) == n_windows

    # Check if targets are inputs shifted by one
    inputs, targets = dataset[0]
    assert (targets[:-1] == inputs[1:]).all() == True

@pytest.mark.parametrize(
    "batch_size, max_length, stride",
    [
        (4, 8, 4),
        (8, 4, 4),
    ]
)
def test_create_data_loader_v1(
    gpt_dataset_args,
    batch_size, max_length, stride
):
    dataloader = create_data_loader_v1(
        gpt_dataset_args['txt'],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    inputs, _ = next(iter(dataloader))
    # Sanity check of dataloader batches
    assert (
        inputs.shape[0] == batch_size and
        inputs.shape[1] == max_length
    )