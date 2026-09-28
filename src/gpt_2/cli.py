import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PretrainArguments:
    config: Path
    input: Path
    output_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt2-pretrain",
        description="Pre-train a GPT-2 model from a local text dataset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML run configuration.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where training artifacts will be written.",
    )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
) -> PretrainArguments:
    namespace = build_parser().parse_args(argv)

    return PretrainArguments(
        config=namespace.config,
        input=namespace.input,
        output_dir=namespace.output_dir,
    )
