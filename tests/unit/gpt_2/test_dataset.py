import math
import urllib.request
from pathlib import Path
import pytest
import tiktoken
from gpt_2.dataset import (
    create_data_loader_v1,
    download_pt_dataset,
    GPTDatasetV1,
)

EXPECTED_DATASET_URL = \
    "https://raw.githubusercontent.com/rasbt/" \
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/" \
    "the-verdict.txt"

@pytest.fixture
def gpt_dataset_args():
    txt = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " \
    "Maecenas sagittis facilisis erat, sit amet fringilla erat eleifend pharetra. " \
    "Integer non metus sagittis, iaculis turpis blandit, gravida erat. " \
    "Curabitur sed suscipit urna, consectetur rhoncus purus. Vestibulum " \
    "scelerisque enim ut nibh vulputate, quis laoreet purus euismod. " \
    "Fusce bibendum elit et mauris ornare, sit amet pellentesque lorem dapibus. " \
    "Vestibulum commodo libero eu libero suscipit malesuada." \
    " Morbi accumsan elit lectus, ut efficitur libero dignissim non."
    tokenizer = tiktoken.get_encoding('gpt2')

    return {'txt': txt, 'tokenizer': tokenizer}

def test_download_dataset_when_destination_is_missing(tmp_path, monkeypatch):
    destination = tmp_path / "the_verdict.txt"
    calls = []

    def fake_retrieve(url, file_path):
        calls.append((url, file_path))
        file_path.write_text("fake dataset", encoding="utf-8")
        return str(file_path), None

    monkeypatch.setattr(urllib.request, 'urlretrieve', fake_retrieve)
    result = download_pt_dataset(destination)

    assert len(calls) == 1
    called_url, called_path = calls[0]

    assert called_url == EXPECTED_DATASET_URL
    assert called_path == destination
    assert result == destination
    assert destination.read_text(encoding="utf-8") == "fake dataset"

def test_reuses_existing_dataset(tmp_path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("urlretrieve should not be called for an existing file")
    monkeypatch.setattr(urllib.request, 'urlretrieve', fail_if_called)

    destination = tmp_path / "the_verdict.txt"
    destination.write_text("existing dataset")
    destination_path = download_pt_dataset(destination)
    assert destination_path == destination

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