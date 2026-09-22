import urllib.request
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

from gpt_2.config import DataConfig

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
        # if stride > self.context_length:
        #     raise ValueError(
        #         f"stride must be smaller or equal than context length"
        #     )

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
        token_ids = tokenizer.encode(txt)
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