import math
import urllib.request
from typing import Sequence

import pytest
import tiktoken
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from gpt_2.config import ModelConfig, DataConfig, TrainingConfig
from gpt_2.dataset import (
    create_data_loader_v1, download_pt_dataset, split_token_ids,
    DataLoaderBundle, GPTTokenDataset, GPTDatasetV1, create_data_loaders,
)

EXPECTED_DATASET_URL = \
    "https://raw.githubusercontent.com/rasbt/" \
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/" \
    "the-verdict.txt"

VALID_MODEL_CONFIG = ModelConfig(
    vocab_size=100,
    context_length=10,
    emb_dim=24,
    n_heads=4,
    n_layers=10,
    drop_rate=0.2,
    qkv_bias=False,
)

VALID_DATASET_CONFIG = DataConfig(
    train_fraction=0.6,
    validation_fraction=0.2,
    stride=2,
    num_workers=0,
)

VALID_TRAINING_CONFIG = TrainingConfig(
    batch_size=10,
    learning_rate=1e-4,
    num_epochs=5,
    eval_every_steps=5,
    eval_batches=10,
    checkpoint_every_steps=10,
    seed=123,
    device='cpu'
)

VALID_GPT_TOKEN_DATASET_PARAMS = {
    "token_ids": range(9),
    "context_length": 4,
    "stride": 2
}

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

def test_split_token_ids_uses_expected_boundaries():
    train, validation, test = split_token_ids(
        list(range(20)),
        VALID_DATASET_CONFIG,
    )

    assert train == list(range(12))
    assert validation == list(range(12, 16))
    assert test == list(range(16, 20))

@pytest.mark.parametrize(
        'token_ids',
        [
            list(range(50)),
            range(60),
            tuple(range(30)),
        ]
)
def test_split_tokens_correctly_split_token_ids(token_ids: Sequence[int]):
    num_train_tokens = int(len(token_ids) * VALID_DATASET_CONFIG.train_fraction)
    num_validation_tokens = int(len(token_ids) * VALID_DATASET_CONFIG.validation_fraction)
    num_test_tokens = len(token_ids) - num_train_tokens - num_validation_tokens
    train_tokens, validation_tokens, test_tokens = \
        split_token_ids(token_ids=token_ids, config=VALID_DATASET_CONFIG)

    assert len(train_tokens) == num_train_tokens
    assert len(validation_tokens) == num_validation_tokens
    assert len(test_tokens) == num_test_tokens

    joined_tokens = list((*train_tokens, *validation_tokens, *test_tokens))
    assert joined_tokens == list(token_ids)

    assert train_tokens == list(token_ids[:num_train_tokens])
    assert validation_tokens == list(
        token_ids[num_train_tokens:num_train_tokens + num_validation_tokens]
    )
    assert test_tokens == list(
        token_ids[num_train_tokens + num_validation_tokens:]
    )

def test_split_tokens_copies_token_ids():
    token_ids = list(range(20))
    original = token_ids.copy()

    train_tokens, _, _ = split_token_ids(
        token_ids,
        VALID_DATASET_CONFIG,
    )

    train_tokens[0] = -1

    assert token_ids == original

def test_split_tokens_rejects_empty_inputs():
    with pytest.raises(ValueError, match='Empty'):
        split_token_ids(token_ids=[], config=VALID_DATASET_CONFIG)

def test_split_tokens_rejects_too_short_inputs():
    with pytest.raises(ValueError, match='Empty'):
        split_token_ids(token_ids=[1,2], config=VALID_DATASET_CONFIG)

def test_split_tokens_handles_sequences():
    train_tokens_iter, validation_tokens_iter, test_tokens_iter = \
        split_token_ids(token_ids=range(1,10), config=VALID_DATASET_CONFIG)
    train_tokens_tuple, validation_tokens_tuple, test_tokens_tuple = \
        split_token_ids(token_ids=tuple(range(1,10)), config=VALID_DATASET_CONFIG)

    assert train_tokens_iter == train_tokens_tuple
    assert validation_tokens_iter == validation_tokens_tuple
    assert test_tokens_iter == test_tokens_tuple

def test_gpt_token_dataset_exact_windows():
    token_ids = list(range(9))
    context_length = 4
    stride=2
    dataset = GPTTokenDataset(
        token_ids=token_ids,
        context_length=context_length,
        stride=stride
    )
    expected_windows = [
        ([0, 1, 2, 3], [1, 2, 3, 4]),
        ([2, 3, 4, 5], [3, 4, 5, 6]),
        ([4, 5, 6, 7], [5, 6, 7, 8]),
    ]

    assert len(dataset) == len(expected_windows)

    for index, (expected_inputs, expected_targets) in enumerate(expected_windows):
        inputs, targets = dataset[index]

        assert inputs.tolist() == expected_inputs
        assert targets.tolist() == expected_targets
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long

@pytest.mark.parametrize(
        'field, value, message',
        [
            ("context_length", 4.0, "context length"),
            ("context_length", True, "context length"),
            ("context_length", 0, "context length"),
            ("context_length", -1, "context length"),
            ("context_length", 50, "context length"),
            ("stride", 4.0, "stride"),
            ("stride", True, "stride"),
            ("stride", 0, "stride"),
            ("stride", -1, "stride"),
        ]
)
def test_gpt_token_dataset_rejects_invalid_params(field, value, message):
    params = VALID_GPT_TOKEN_DATASET_PARAMS | {field : value}
    with pytest.raises(ValueError, match=message):
        GPTTokenDataset(**params)

def test_gpt_token_dataset_too_few_tokens():
    token_ids = list(range(3))
    context_length = 4
    stride=1

    with pytest.raises(ValueError, match="length"):
        GPTTokenDataset(
            token_ids=token_ids,
            context_length=context_length,
            stride=stride
        )

def test_gpt_token_dataset_one_window():
    token_ids = list(range(5))
    context_length = 4
    stride=1
    dataset = GPTTokenDataset(
        token_ids=token_ids,
        context_length=context_length,
        stride=stride
    )
    assert dataset.n_windows == 1

def test_gpt_token_dataset_and_dataset_v1_produce_same_tokens():
    txt = "hello this is a test of the tokenizer"
    tokenizer = tiktoken.get_encoding('gpt2')

    token_ids = tokenizer.encode(txt)
    context_length = 4
    stride = 2

    token_dataset = GPTTokenDataset(
        token_ids=token_ids,
        context_length=context_length,
        stride=stride,
    )
    v1_dataset = GPTDatasetV1(
        txt=txt, tokenizer=tokenizer,
        max_length=context_length, stride=stride,
    )

    assert len(token_dataset) == len(v1_dataset)

    for index in range(len(token_dataset)):
        token_inputs, token_targets = token_dataset[index]
        v1_inputs, v1_targets = v1_dataset[index]

        assert torch.equal(token_inputs, v1_inputs)
        assert torch.equal(token_targets, v1_targets)

def test_gpt_token_independent_tensor_memory():
    original = list(range(9))
    copy = original.copy()
    context_length = 4
    stride = 2

    dataset = GPTTokenDataset(
        token_ids=original,
        context_length=context_length,
        stride=stride
    )

    inputs, targets = dataset[0]
    inputs[0] = -1
    targets[0] = -1

    assert original == copy

def test_data_loaders_contain_correct_split():
    token_ids = list(range(101))
    loaders = create_data_loaders(
        token_ids=token_ids,
        model_config=VALID_MODEL_CONFIG,
        training_config=VALID_TRAINING_CONFIG,
        data_config=VALID_DATASET_CONFIG,
    )

    train_end = int(len(token_ids) * VALID_DATASET_CONFIG.train_fraction)
    validation_size = int(
        len(token_ids) * VALID_DATASET_CONFIG.validation_fraction
    )
    validation_end = train_end + validation_size

    train_dataset = loaders.train.dataset
    validation_dataset = loaders.validation.dataset
    test_dataset = loaders.test.dataset

    assert isinstance(loaders, DataLoaderBundle)
    assert isinstance(train_dataset, GPTTokenDataset)
    assert isinstance(validation_dataset, GPTTokenDataset)
    assert isinstance(test_dataset, GPTTokenDataset)
    assert train_dataset.token_ids.tolist() == token_ids[:train_end]
    assert validation_dataset.token_ids.tolist() == token_ids[
        train_end:validation_end
    ]
    assert test_dataset.token_ids.tolist() == token_ids[validation_end:]


def test_data_loaders_use_configured_batch_size_and_workers():
    token_ids = list(range(101))

    loaders = create_data_loaders(
        token_ids=token_ids,
        model_config=VALID_MODEL_CONFIG,
        training_config=VALID_TRAINING_CONFIG,
        data_config=VALID_DATASET_CONFIG,
    )

    for loader in (loaders.train, loaders.validation, loaders.test):
        assert loader.batch_size == VALID_TRAINING_CONFIG.batch_size
        assert loader.num_workers == VALID_DATASET_CONFIG.num_workers


def test_data_loaders_use_correct_samplers():
    token_ids = list(range(101))
    loaders = create_data_loaders(
        token_ids=token_ids,
        model_config=VALID_MODEL_CONFIG,
        training_config=VALID_TRAINING_CONFIG,
        data_config=VALID_DATASET_CONFIG,
    )

    assert isinstance(loaders.train.sampler, RandomSampler)
    assert isinstance(loaders.validation.sampler, SequentialSampler)
    assert isinstance(loaders.test.sampler, SequentialSampler)


def test_data_loaders_retain_partial_final_batches():
    token_ids = list(range(132))
    loaders = create_data_loaders(
        token_ids=token_ids,
        model_config=VALID_MODEL_CONFIG,
        training_config=VALID_TRAINING_CONFIG,
        data_config=VALID_DATASET_CONFIG,
    )

    for loader in (loaders.train, loaders.validation, loaders.test):
        batches = list(loader)
        loaded_samples = sum(inputs.shape[0] for inputs, _ in batches)
        dataset = loader.dataset
        assert isinstance(dataset, GPTTokenDataset)
        dataset_size = len(dataset)
        remainder = dataset_size % VALID_TRAINING_CONFIG.batch_size

        assert remainder > 0
        assert loaded_samples == dataset_size
        assert batches[-1][0].shape[0] == remainder
        assert loader.drop_last is False


def test_data_loaders_same_seed_produce_same_first_epoch():
    token_ids = list(range(210))
    first = create_data_loaders(
        token_ids=token_ids,
        model_config=VALID_MODEL_CONFIG,
        data_config=VALID_DATASET_CONFIG,
        training_config=VALID_TRAINING_CONFIG,
    )
    second = create_data_loaders(
        token_ids=token_ids,
        model_config=VALID_MODEL_CONFIG,
        data_config=VALID_DATASET_CONFIG,
        training_config=VALID_TRAINING_CONFIG,
    )

    first_batches = list(first.train)
    second_batches = list(second.train)
    first_epoch_inputs = torch.cat([inputs for inputs, _ in first_batches])
    second_epoch_inputs = torch.cat([inputs for inputs, _ in second_batches])
    first_epoch_targets = torch.cat([targets for _, targets in first_batches])
    second_epoch_targets = torch.cat([targets for _, targets in second_batches])

    assert len(first_batches) > 1
    assert torch.equal(first_epoch_inputs, second_epoch_inputs)
    assert torch.equal(first_epoch_targets, second_epoch_targets)


def test_evaluation_data_loaders_preserve_window_order():
    token_ids = list(range(101))
    loaders = create_data_loaders(
        token_ids=token_ids,
        model_config=VALID_MODEL_CONFIG,
        data_config=VALID_DATASET_CONFIG,
        training_config=VALID_TRAINING_CONFIG,
    )

    validation_starts = torch.cat(
        [inputs[:, 0] for inputs, _ in loaders.validation]
    ).tolist()
    test_starts = torch.cat(
        [inputs[:, 0] for inputs, _ in loaders.test]
    ).tolist()

    assert validation_starts == [60, 62, 64, 66, 68]
    assert test_starts == [80, 82, 84, 86, 88, 90]


def test_data_loaders_reject_too_short_split():
    token_ids = list(range(20))
    with pytest.raises(ValueError, match="validation"):
        create_data_loaders(
            token_ids=token_ids,
            model_config=VALID_MODEL_CONFIG,
            training_config=VALID_TRAINING_CONFIG,
            data_config=VALID_DATASET_CONFIG,
        )


def test_data_loaders_produce_expected_shape_and_dtype():
    token_ids = list(range(101))
    loaders = create_data_loaders(
        token_ids=token_ids,
        model_config=VALID_MODEL_CONFIG,
        training_config=VALID_TRAINING_CONFIG,
        data_config=VALID_DATASET_CONFIG,
    )

    for loader in (loaders.train, loaders.validation, loaders.test):
        for inputs, targets in loader:
            assert inputs.shape[-1] == VALID_MODEL_CONFIG.context_length
            assert targets.shape[-1] == VALID_MODEL_CONFIG.context_length
            assert inputs.dtype == torch.long
            assert targets.dtype == torch.long
