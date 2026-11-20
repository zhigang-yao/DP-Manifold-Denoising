"""Observational-noise samplers, separate from all privacy mechanisms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def sample_uniform_ball_noise(
    count: int,
    dimension: int,
    radius: float,
    rng: np.random.Generator,
) -> Array:
    """Draw row-oriented noise uniformly from a Euclidean ball.

    This is bounded observational noise. It is not DP Gaussian noise, GP
    release noise, or any other privacy perturbation.
    """

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, (int, np.integer))
        or dimension < 1
    ):
        raise ValueError("dimension must be a positive integer")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and nonnegative")
    count = int(count)
    dimension = int(dimension)
    if count == 0 or radius == 0.0:
        return np.zeros((count, dimension), dtype=float)

    gaussian = np.asarray(rng.standard_normal((count, dimension)), dtype=float)
    norms = np.linalg.norm(gaussian, axis=1)
    while np.any(norms == 0.0):
        zero = norms == 0.0
        gaussian[zero] = rng.standard_normal((int(np.sum(zero)), dimension))
        norms[zero] = np.linalg.norm(gaussian[zero], axis=1)
    directions = gaussian / norms[:, None]
    radial = radius * rng.random(count) ** (1.0 / dimension)
    return directions * radial[:, None]
