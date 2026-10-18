import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from gpt_2.config import DataConfig, RunConfig
from gpt_2.dataset import split_token_ids
from gpt_2.runtime import ResolvedRuntime


def build_run_metadata(
    config: RunConfig,
    runtime: ResolvedRuntime,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "config": asdict(config),
        "runtime": asdict(runtime) | {"device": str(runtime.device)},
    }


def write_run_metadata(
    path: Path, *, config: RunConfig, runtime: ResolvedRuntime
) -> None:
    metadata = build_run_metadata(
        config=config,
        runtime=runtime,
    )
    serialized_metadata = json.dumps(metadata, indent=2, allow_nan=False) + "\n"

    with open(path, mode="x", encoding="utf-8") as fp:
        fp.write(serialized_metadata)


def build_dataset_metadata(
    source_bytes: bytes,
    token_ids: Sequence[int],
    config: DataConfig,
) -> dict[str, object]:
    train, valid, test = split_token_ids(token_ids=token_ids, config=config)
    start_train, end_train = 0, len(train)
    start_valid, end_valid = end_train, end_train + len(valid)
    start_test, end_test = end_valid, end_valid + len(test)

    return {
        "schema_version": 1,
        "source": {
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "size_bytes": len(source_bytes),
        },
        "tokenizer": {
            "name": "gpt2",
            "special_token_policy": "ordinary_text",
        },
        "num_tokens": len(token_ids),
        "splits": {
            "train": {
                "start_token": start_train,
                "end_token": end_train,
                "num_tokens": len(train),
            },
            "validation": {
                "start_token": start_valid,
                "end_token": end_valid,
                "num_tokens": len(valid),
            },
            "test": {
                "start_token": start_test,
                "end_token": end_test,
                "num_tokens": len(test),
            },
        },
    }
