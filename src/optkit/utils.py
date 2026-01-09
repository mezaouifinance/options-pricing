from __future__ import annotations
import math
from dataclasses import dataclass

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)

def norm_cdf(x: float) -> float:
    # Standard normal CDF via erf (stable enough for finance usage)
    return 0.5 * (1.0 + math.erf(x / SQRT_2))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def discount_factor(r: float, T: float) -> float:
    return math.exp(-r * T)

def assert_positive(name: str, x: float, allow_zero: bool = False) -> None:
    if allow_zero:
        if x < 0:
            raise ValueError(f"{name} must be >= 0")
    else:
        if x <= 0:
            raise ValueError(f"{name} must be > 0")

@dataclass(frozen=True)
class Market:
    S: float
    r: float

@dataclass(frozen=True)
class Contract:
    K: float
    T: float
    option_type: str

