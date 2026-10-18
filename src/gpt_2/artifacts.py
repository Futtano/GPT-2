import json
from dataclasses import asdict
from pathlib import Path

from gpt_2.config import RunConfig
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
