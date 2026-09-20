import math
from numbers import Real
from typing import Any
from collections.abc import Mapping
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool

    def __post_init__(self) -> None:
        # Sanity checks
        if not isinstance(self.vocab_size, int) or isinstance(self.vocab_size, bool):
            raise ValueError(f"{self.vocab_size=} must be an integer.")
        if self.vocab_size <= 0:
            raise ValueError(f"{self.vocab_size=} must be greater than zero.")
        if not isinstance(self.context_length, int) or isinstance(self.context_length, bool):
            raise ValueError(f"{self.context_length=} must be an integer.")
        if self.context_length <= 0:
            raise ValueError(f"{self.context_length=} must be greater than zero.")
        if not isinstance(self.emb_dim, int) or isinstance(self.emb_dim, bool):
            raise ValueError(f"{self.emb_dim=} must be an integer.")
        if self.emb_dim <= 0:
            raise ValueError(f"{self.emb_dim=} must be greater than zero.")
        if not isinstance(self.n_heads, int) or isinstance(self.n_heads, bool):
            raise ValueError(f"{self.n_heads=} must be an integer.")
        if self.n_heads <= 0:
            raise ValueError(f"{self.n_heads=} must be greater than zero.")
        if not isinstance(self.n_layers, int) or isinstance(self.n_layers, bool):
            raise ValueError(f"{self.n_layers=} must be an integer.")
        if self.n_layers <= 0:
            raise ValueError(f"{self.n_layers=} must be greater than zero.")
        if isinstance(self.drop_rate, bool) or not isinstance(self.drop_rate, Real):
            raise ValueError(f"{self.drop_rate=} must be a real number")
        if self.drop_rate < 0.0 or self.drop_rate > 1.0 or not math.isfinite(self.drop_rate):
            raise ValueError(
                f"{self.drop_rate=} must be a quantity between 0.0 (inclusive) and 1.0 (inclusive)."
            )
        if self.qkv_bias is not True and self.qkv_bias is not False:
            raise ValueError(f"{self.qkv_bias=} must be either True or False.")

        # emb_dim must be divisible by n_heads
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(f"{self.emb_dim=} must be divisible by {self.n_heads=}.")

def ensure_model_config(config: ModelConfig | Mapping[str, Any]) -> ModelConfig:
    if isinstance(config, ModelConfig):
        return config

    return ModelConfig(**config)