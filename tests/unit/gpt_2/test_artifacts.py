import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gpt_2.artifacts import (
    build_dataset_metadata,
    build_run_metadata,
    write_run_metadata,
)
from gpt_2.config import DataConfig, RunConfig
from gpt_2.runtime import ResolvedRuntime


@pytest.fixture
def invalid_file_path(tmp_path: Path) -> Path:
    return tmp_path / "foo" / "metadata.json"


@pytest.fixture
def expected_string():
    return "Hello World!"


@pytest.fixture
def existing_file(tmp_path: Path, expected_string: str) -> Path:
    file = tmp_path / "metadata.json"
    file.write_text(expected_string, encoding="utf-8")
    return file


@pytest.fixture
def file_name(tmp_path: Path) -> Path:
    file = tmp_path / "metadata.json"
    return file


@pytest.fixture
def raw_bytes():
    return b"Hello \xff\xfe\x41" * 10


@pytest.fixture
def token_ids():
    return range(100)


def test_build_run_metadata(
    run_config: RunConfig,
    runtime: ResolvedRuntime,
):
    metadata = build_run_metadata(
        config=run_config,
        runtime=runtime,
    )

    assert metadata == {
        "schema_version": 1,
        "config": asdict(run_config),
        "runtime": {
            "device": str(runtime.device),
            "seed": runtime.seed,
        },
    }


def test_build_run_metadata_requested_auto(
    run_config: RunConfig,
    runtime: ResolvedRuntime,
):
    run_config = replace(
        run_config, training=replace(run_config.training, device="auto")
    )
    metadata = build_run_metadata(
        config=run_config,
        runtime=runtime,
    )

    assert metadata == {
        "schema_version": 1,
        "config": asdict(run_config),
        "runtime": {
            "device": str(runtime.device),
            "seed": runtime.seed,
        },
    }


def test_write_metadata(
    file_name: Path, run_config: RunConfig, runtime: ResolvedRuntime
):
    expected = build_run_metadata(
        config=run_config,
        runtime=runtime,
    )

    write_run_metadata(file_name, config=run_config, runtime=runtime)
    with open(file_name, encoding="utf-8") as fp:
        metadata = json.load(fp)

    assert metadata == expected


def test_write_metadata_rejects_existing_file(
    existing_file: Path,
    expected_string: str,
    run_config: RunConfig,
    runtime: ResolvedRuntime,
):
    with pytest.raises(FileExistsError):
        write_run_metadata(existing_file, config=run_config, runtime=runtime)

    content = existing_file.read_text(encoding="utf-8")
    assert content == expected_string


def test_write_metadata_raises_if_parent_dir_not_exists(
    invalid_file_path: Path,
    run_config: RunConfig,
    runtime: ResolvedRuntime,
):
    with pytest.raises(FileNotFoundError):
        write_run_metadata(invalid_file_path, config=run_config, runtime=runtime)


def test_metadata_handlers_do_not_change_originals(
    file_name: Path,
    run_config: RunConfig,
    runtime: ResolvedRuntime,
):
    copy_run_config = copy.deepcopy(run_config)
    copy_runtime = copy.deepcopy(runtime)
    _ = build_run_metadata(
        config=run_config,
        runtime=runtime,
    )

    assert copy_run_config == run_config
    assert copy_runtime == runtime

    write_run_metadata(file_name, config=run_config, runtime=runtime)

    assert copy_run_config == run_config
    assert copy_runtime == runtime


@pytest.mark.parametrize("token_ids", [range(7), list(range(13)), tuple(range(100))])
def test_build_dataset_metadata(
    raw_bytes: bytes,
    token_ids: Sequence[int],
    data_config: DataConfig,
):
    metadata = build_dataset_metadata(
        source_bytes=raw_bytes,
        token_ids=token_ids,
        config=data_config,
    )

    start_train, end_train = 0, int(len(token_ids) * data_config.train_fraction)
    start_valid, end_valid = (
        end_train,
        end_train + int(len(token_ids) * data_config.validation_fraction),
    )
    start_test, end_test = end_valid, len(token_ids)

    assert metadata == {
        "schema_version": 1,
        "source": {
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "size_bytes": len(raw_bytes),
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
                "num_tokens": end_train,
            },
            "validation": {
                "start_token": start_valid,
                "end_token": end_valid,
                "num_tokens": end_valid - start_valid,
            },
            "test": {
                "start_token": start_test,
                "end_token": end_test,
                "num_tokens": end_test - start_test,
            },
        },
    }


def test_build_dataset_metadata_hash_change(
    raw_bytes: bytes,
    token_ids: Sequence[int],
    data_config: DataConfig,
):
    metadata = build_dataset_metadata(
        source_bytes=raw_bytes,
        token_ids=token_ids,
        config=data_config,
    )

    changed_bytes = b"ciao"
    changed_metadata = build_dataset_metadata(
        source_bytes=changed_bytes,
        token_ids=token_ids,
        config=data_config,
    )

    assert changed_metadata == metadata | {
        "source": {
            "sha256": hashlib.sha256(changed_bytes).hexdigest(),
            "size_bytes": len(changed_bytes),
        }
    }
    original_source = metadata["source"]
    changed_source = changed_metadata["source"]
    assert isinstance(original_source, dict)
    assert isinstance(changed_source, dict)
    assert original_source["sha256"] != changed_source["sha256"]


def test_build_dataset_metadata_rounding(
    raw_bytes: bytes,
    data_config: DataConfig,
):
    metadata = build_dataset_metadata(
        source_bytes=raw_bytes,
        token_ids=range(7),
        config=data_config,
    )

    assert metadata == {
        "schema_version": 1,
        "source": {
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "size_bytes": len(raw_bytes),
        },
        "tokenizer": {
            "name": "gpt2",
            "special_token_policy": "ordinary_text",
        },
        "num_tokens": 7,
        "splits": {
            "train": {
                "start_token": 0,
                "end_token": 4,
                "num_tokens": 4,
            },
            "validation": {
                "start_token": 4,
                "end_token": 5,
                "num_tokens": 1,
            },
            "test": {
                "start_token": 5,
                "end_token": 7,
                "num_tokens": 2,
            },
        },
    }


@pytest.mark.parametrize(
    "token_count, train_fraction, validation_fraction",
    [
        (0, 0.6, 0.2),
        (1, 0.6, 0.2),
        (3, 0.1, 0.4),
        (3, 0.6, 0.1),
    ],
)
def test_build_dataset_metadata_rejects_empty_splits(
    raw_bytes: bytes,
    data_config: DataConfig,
    token_count: int,
    train_fraction: float,
    validation_fraction: float,
):
    data_config = replace(
        data_config,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    with pytest.raises(ValueError, match="Empty splits"):
        build_dataset_metadata(
            source_bytes=raw_bytes,
            token_ids=range(token_count),
            config=data_config,
        )


@pytest.mark.parametrize("token_count", [7, 13, 100])
def test_build_dataset_metadata_boundaries_cover_all_tokens(
    raw_bytes: bytes,
    data_config: DataConfig,
    token_count: int,
):
    token_ids = list(range(token_count))
    metadata = build_dataset_metadata(raw_bytes, token_ids, data_config)
    splits = metadata["splits"]
    assert isinstance(splits, dict)

    cursor = 0
    reconstructed_tokens = []
    for name in ("train", "validation", "test"):
        split = splits[name]
        assert isinstance(split, dict)
        start = split["start_token"]
        end = split["end_token"]
        assert start == cursor
        assert end > start
        assert split["num_tokens"] == end - start
        reconstructed_tokens.extend(token_ids[start:end])
        cursor = end

    assert cursor == metadata["num_tokens"] == token_count
    assert reconstructed_tokens == token_ids


def test_build_dataset_metadata_preserves_input_tokens(
    raw_bytes: bytes,
    data_config: DataConfig,
):
    token_ids = [8, 3, 5, 2, 9, 1, 4]
    original_tokens = token_ids.copy()
    original_config = asdict(data_config)

    build_dataset_metadata(raw_bytes, token_ids, data_config)

    assert token_ids == original_tokens
    assert asdict(data_config) == original_config
