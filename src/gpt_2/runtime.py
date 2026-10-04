import random
from dataclasses import dataclass

import numpy as np
import torch

from gpt_2.config import DeviceName


def resolve_device(name: DeviceName) -> torch.device:
    match name:
        case "cuda":
            if torch.cuda.is_available():
                return torch.device(name)
            raise RuntimeError(f"accelerator {name} is not available on your machine.")
        case "mps":
            if torch.backends.mps.is_available():
                return torch.device(name)
            raise RuntimeError(f"accelerator {name} is not available on your machine.")
        case "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        case "cpu":
            return torch.device("cpu")
        case _:
            raise RuntimeError(f"device {name} is currently not supported.")


def seed_random_sources(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class ResolvedRuntime:
    device: torch.device
    seed: int
