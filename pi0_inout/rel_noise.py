"""
rel_noise.py
---------------------------
Utilities for injecting relative-error noise into matmul outputs.

Two noise modes are supported:

  "uniform"  — ±rel_err * |y| with random sign per element.
      Every element gets the full perturbation magnitude; only the sign is
      random.  This is a two-point distribution (Rademacher * rel_err),
      NOT a continuous uniform.  It's the most aggressive choice: every
      element is always perturbed by the maximum amount.

  "laplace"  — Laplace-distributed noise scaled by |y|.
      Each element is perturbed by Laplace(0, rel_err) * |y|.  Empirical
      testing (experiments/test_error_distribution.py) with KS goodness-of-fit
      showed that:
        - Element-wise FP8 quantization error (before matmul) is slightly
          better fit by Gaussian (wins 5/5 in Part 1).
        - After matmul propagation, the error becomes sharply peaked with
          heavy tails: Laplace wins 5/5 across dims 32-1024 (Part 2,
          KS ~0.085 vs ~0.21 for Gaussian).
        - Through chained layers, Laplace wins 6/6 (Part 3), and the
          advantage grows with depth as kurtosis trends toward Laplace's
          theoretical value of 3.
      Since noise injection operates on matmul outputs (not raw element
      quantization), Laplace is the appropriate fit: it wins 11/11 on
      the matmul-output cases that this function targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class NoiseMode(str, Enum):
    UNIFORM = "uniform"
    LAPLACE = "laplace"


@dataclass(frozen=True)
class RelNoiseConfig:
    """Relative-error noise: inject noise scaled by |y| into each matmul output element."""
    rel_err: float = 0.0
    mode: NoiseMode = NoiseMode.UNIFORM

    def enabled(self) -> bool:
        return self.rel_err != 0.0


def inject_rel_noise(
    y: torch.Tensor,
    *,
    rel_err: float,
    mode: NoiseMode = NoiseMode.UNIFORM,
) -> torch.Tensor:
    """
    Add relative-error noise to y.

    Args:
        y:        Tensor to perturb (typically a matmul output).
        rel_err:  Noise scale as a dimensionless fraction (e.g. 0.01 = 1%).
        mode:     "uniform" — ±rel_err * |y| (two-point / Rademacher).
                  "laplace" — Laplace(0, rel_err) * |y|.
    """
    if rel_err == 0.0:
        return y
    if rel_err < 0:
        raise ValueError("rel_err must be >= 0")

    if mode == NoiseMode.UNIFORM:
        sign = torch.where(
            torch.rand_like(y) < 0.5,
            torch.tensor(-1.0, device=y.device, dtype=y.dtype),
            torch.tensor(1.0, device=y.device, dtype=y.dtype),
        )
        return y + rel_err * y.abs() * sign

    elif mode == NoiseMode.LAPLACE:
        # Laplace(0, b) = -b * sign(U) * ln(1 - 2|U|) where U ~ Uniform(-0.5, 0.5)
        # Equivalently: sample from Exponential(1/b) with random sign.
        # torch.distributions is slow; manual inverse-CDF is faster.
        u = torch.rand_like(y) - 0.5  # U ~ Uniform(-0.5, 0.5)
        # Clamp to avoid log(0)
        laplace = -rel_err * u.sign() * torch.log1p(-2.0 * u.abs()).clamp(min=-20.0)
        return y + laplace * y.abs()

    else:
        raise ValueError(f"Unknown noise mode: {mode}")
