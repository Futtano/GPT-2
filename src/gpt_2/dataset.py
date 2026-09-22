import urllib.request
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

from gpt_2.config import DataConfig

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i+max_length]
            target_chunk = token_ids[i+1 : i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

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
            f"total token count is {len(token_ids)}, "
            f"training token count is {len(train_tokens)}, "
            f"validation token count is {len(validation_tokens)}, "
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