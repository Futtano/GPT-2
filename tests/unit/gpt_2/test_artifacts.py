import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gpt_2.artifacts import build_run_metadata, write_run_metadata
from gpt_2.config import RunConfig
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
