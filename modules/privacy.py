"""Shared analytic-Gaussian calibration for canonical DP comparators."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, log_ndtr, ndtr


def gaussian_dp_delta(epsilon: float, sigma: float, sensitivity: float) -> float:
    if epsilon <= 0 or sensitivity < 0:
        raise ValueError("invalid Gaussian privacy parameters")
    if sensitivity == 0:
        return 0.0
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    left = sensitivity / (2.0 * sigma) - epsilon * sigma / sensitivity
    right = -sensitivity / (2.0 * sigma) - epsilon * sigma / sensitivity
    log_second = epsilon + float(log_ndtr(right))
    second = (
        0.0 if log_second < np.log(np.finfo(float).tiny) else float(np.exp(log_second))
    )
    return float(np.clip(ndtr(left) - second, 0.0, 1.0))


def analytic_gaussian_sigma(
    epsilon: float,
    delta: float,
    sensitivity: float,
    *,
    tolerance: float = 1e-12,
) -> tuple[float, float]:
    if epsilon <= 0 or not 0 < delta < 1 or sensitivity < 0:
        raise ValueError("invalid analytic Gaussian parameters")
    if sensitivity == 0:
        return 0.0, 0.0
    lower, upper = 0.0, 1.0
    while gaussian_dp_delta(epsilon, upper, 1.0) > delta:
        upper *= 2.0
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        if gaussian_dp_delta(epsilon, midpoint, 1.0) > delta:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance * max(1.0, upper):
            break
    sigma = sensitivity * upper
    return sigma, gaussian_dp_delta(epsilon, sigma, sensitivity)


def chi_mean(dimension: int) -> float:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return float(
        np.sqrt(2.0)
        * np.exp(gammaln((dimension + 1.0) / 2.0) - gammaln(dimension / 2.0))
    )
