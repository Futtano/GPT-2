import math
from numbers import Real
from typing import Any, Literal
from collections.abc import Mapping
from dataclasses import dataclass

def _validate_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

def _validate_nonnegative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")

def _validate_finite_bounded_float(
    name: str,
    value: object,
    lower: float,
    upper: float,
    inclusive_lower: bool,
    inclusive_upper: bool,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number.")

    try:
        numeric_value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be representable as a finite float.") from error

    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")

    below_lower = numeric_value < lower
    above_upper = numeric_value > upper

    if below_lower or (numeric_value == lower and not inclusive_lower):
        raise ValueError(f"{name} is below its permitted range.")

    if above_upper or (numeric_value == upper and not inclusive_upper):
        raise ValueError(f"{name} is above its permitted range.")

def _validate_boolean(name: str, value: object) -> None:
    if value is not True and value is not False:
        raise ValueError(f"{name} must be either True or False.")

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
        _validate_positive_int("vocab_size", self.vocab_size)
        _validate_positive_int("context_length", self.context_length)
        _validate_positive_int("emb_dim", self.emb_dim)
        _validate_positive_int("n_heads", self.n_heads)
        _validate_positive_int("n_layers", self.n_layers)
        _validate_finite_bounded_float(
            'drop_rate', self.drop_rate,
            lower=0.0, upper=1.0,
            inclusive_lower=True, inclusive_upper=True
        )

        _validate_boolean("qkv_bias", self.qkv_bias)

        # emb_dim must be divisible by n_heads
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(f"{self.emb_dim=} must be divisible by {self.n_heads=}.")

DeviceName = Literal['auto', 'cpu', 'cuda', 'mps']
ALLOWED_DEVICES = ("auto", "cpu", "cuda", "mps")

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    eval_every_steps: int
    eval_batches: int
    checkpoint_every_steps: int
    seed: int
    device: DeviceName

    def __post_init__(self) -> None:
        # Sanity checks
        _validate_positive_int('batch_size', self.batch_size)
        _validate_positive_int('num_epochs', self.num_epochs)
        _validate_positive_int('eval_every_steps', self.eval_every_steps)
        _validate_positive_int('eval_batches', self.eval_batches)
        _validate_positive_int('checkpoint_every_steps', self.checkpoint_every_steps)
        _validate_nonnegative_int('seed', self.seed)
        _validate_finite_bounded_float(
            'learning_rate', self.learning_rate,
            lower=0.0, upper=float('inf'),
            inclusive_lower=False, inclusive_upper=False
        )

        if self.device not in ALLOWED_DEVICES:
            raise ValueError(f"device must be one of {ALLOWED_DEVICES}; got {self.device!r}.")

def ensure_model_config(config: ModelConfig | Mapping[str, Any]) -> ModelConfig:
    if isinstance(config, ModelConfig):
        return config

    return ModelConfig(**config)