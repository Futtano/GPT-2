import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

from gpt_2.config import ModelConfig, DataConfig, TrainingConfig

class GPTTokenDataset(Dataset):
    def __init__(
            self, token_ids: Sequence[int], context_length: int,
            stride: int
    ) -> None:
        super().__init__()

        if not isinstance(context_length, int) or isinstance(context_length, bool):
            raise ValueError(f"context length must be an integer quantity")
        if context_length <= 0:
            raise ValueError("context length must be strictly bigger than zero.")
        if context_length > len(token_ids) - 1:
            raise ValueError(
                f"number of dataset tokens ({len(token_ids)})\n" \
                f"is not sufficient for a context length of size {context_length}."
            )
        if not isinstance(stride, int) or isinstance(stride, bool):
            raise ValueError(f"stride must be an integer quantity")
        if stride <= 0:
            raise ValueError("stride must be strictly bigger than zero.")
        # len(token_ids) - context_length - 1 is the last window's start index
        # divide by stride and add 1 (count the window starting at zero) to
        # obtain the total number of windows
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride
        self.n_windows = (
            (len(token_ids) - context_length - 1) // stride
        ) + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        stop = start + self.context_length

        inputs = self.token_ids[start:stop]
        targets = self.token_ids[start + 1:stop + 1]

        return inputs, targets

    def __len__(self) -> int:
        return self.n_windows

class GPTDatasetV1(GPTTokenDataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        token_ids: list[int] = tokenizer.encode(txt)
        super().__init__(
            token_ids=token_ids,
            context_length=max_length,
            stride=stride,
        )

def split_token_ids(
        token_ids: Sequence[int],
        config: DataConfig,
) -> tuple[list[int], list[int], list[int]]:
    train_size = int(len(token_ids) * config.train_fraction)
    validation_size = int(len(token_ids) * config.validation_fraction)

    train_end = train_size
    validation_end = train_end + validation_size

    train_tokens = list(token_ids[: train_end])
    validation_tokens = list(token_ids[train_end: validation_end])
    test_tokens = list(token_ids[validation_end:])

    is_any_split_empty =  not len(train_tokens) or not len(validation_tokens) or not len(test_tokens)
    if is_any_split_empty:
        raise ValueError(
            "Empty splits are not allowed: "
            f"total token count is {len(token_ids)},\n"
            f"training token count is {len(train_tokens)},\n "
            f"validation token count is {len(validation_tokens)},\n "
            f"test token count is {len(test_tokens)}."
        )

    return train_tokens, validation_tokens, test_tokens

@dataclass(frozen=True)
class DataLoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader

def create_data_loaders(
    token_ids: Sequence[int],
    *,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
) -> DataLoaderBundle:
    train_split, validation_split, test_split = \
        split_token_ids(token_ids=token_ids, config=data_config)

    if len(train_split) < model_config.context_length + 1:
        raise ValueError(
            f"train split with {len(train_split)} tokens is too\n"
            f"short to form a single input-target window of size {model_config.context_length + 1}"
        )
    if len(validation_split) < model_config.context_length + 1:
        raise ValueError(
                    f"validation split with {len(validation_split)} tokens is too\n"
                    f"short to form a single input-target window of size {model_config.context_length + 1}"
                )
    if len(test_split) < model_config.context_length + 1:
        raise ValueError(
                    f"test split with {len(test_split)} tokens is too\n"
                    f"short to form a single input-target window of size {model_config.context_length + 1}"
                )

    train_dataset = GPTTokenDataset(
        token_ids=train_split, context_length=model_config.context_length,
        stride=data_config.stride,
    )
    validation_dataset = GPTTokenDataset(
        token_ids=validation_split, context_length=model_config.context_length,
        stride=data_config.stride,
    )
    test_dataset = GPTTokenDataset(
        token_ids=test_split, context_length=model_config.context_length,
        stride=data_config.stride,
    )

    generator = torch.Generator()
    generator.manual_seed(training_config.seed)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=data_config.num_workers,
        drop_last=False,
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        drop_last=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        drop_last=False,
    )

    return DataLoaderBundle(
        train=train_dataloader,
        validation=validation_dataloader,
        test=test_dataloader,
    )

def create_data_loader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(
        txt, tokenizer, max_length, stride
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader

def download_pt_dataset(file_path="data/the-verdict.txt"):
    if Path(file_path).exists():
        return file_path

    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    urllib.request.urlretrieve(url, file_path)
    return file_path

def get_root():
    return Path.cwd()
