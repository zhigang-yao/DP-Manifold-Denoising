"""Versioned, public estimator settings. No ground truth or risk enters selection."""
from dataclasses import dataclass, asdict, replace
from typing import Literal
import numpy as np
from modules.md.shared_memory import validate_budget

DEFAULT_HS = tuple(float(0.025 * 2 ** (j / 2)) for j in range(9))

@dataclass(frozen=True)
class MethodConfig:
    backend: Literal["auto", "joint", "per_scale", "structured_shared", "structured_independent",
                     "structured_per_query"] = "auto"
    bandwidths: tuple[float, ...] = DEFAULT_HS
    alpha0: float = 0.1
    inner_B: int = 100
    inner_estimator: Literal["mc", "frobenius_cv"] = "mc"
    outer_R: int = 100
    auxiliary_chunk_size: int = 16
    quadrature_order: int = 128
    pointwise_fallback: bool = False
    gate_threshold: float | None = None
    psd_solver: Literal["auto", "dense", "tridiagonal"] = "auto"
    psd_tol: float = 1e-10
    psd_chunk_size: int = 128
    memory_limit_gib: float = 8.0
    auxiliary_seed: int = 918337
    auxiliary_noise_model: Literal["matched","shared_marginal"] = "matched"

    def resolved(self, D: int):
        if D < 2:
            raise ValueError("ambient dimension must be >=2")
        chosen = ("joint" if D <= 4 else "structured_shared") if self.backend == "auto" else self.backend
        if chosen == "joint" and D > 4:
            raise ValueError("joint finite-spectrum backend validated only for D<=4; choose structured_shared")
        return replace(self, backend=chosen)

    def validate(self, D: int):
        c = self.resolved(D)
        validate_budget(c.memory_limit_gib)
        h = np.asarray(c.bandwidths, float)
        if h.ndim != 1 or not len(h) or not np.all(np.isfinite(h)) or np.any(h <= 0) or np.any(np.diff(h) <= 0):
            raise ValueError("strictly increasing positive public bandwidths required")
        if not 0 < c.alpha0 < 1:
            raise ValueError("alpha0 must lie in (0,1)")
        if (isinstance(c.inner_B, bool) or c.inner_B < 1
                or isinstance(c.outer_R, bool) or c.outer_R < 2
                or isinstance(c.auxiliary_chunk_size, bool) or not isinstance(c.auxiliary_chunk_size, int)
                or c.auxiliary_chunk_size < 1
                or isinstance(c.quadrature_order, bool) or c.quadrature_order < 8
                or c.psd_tol <= 0 or c.psd_chunk_size < 1 or c.memory_limit_gib <= 0):
            raise ValueError("invalid public numeric configuration")
        if c.auxiliary_noise_model not in ("matched","shared_marginal"):
            raise ValueError("auxiliary_noise_model must be matched or shared_marginal")
        if c.inner_estimator not in ("mc", "frobenius_cv"):
            raise ValueError("inner_estimator must be mc or frobenius_cv")
        if c.auxiliary_noise_model=="shared_marginal" and c.backend not in (
                "per_scale","structured_independent"):
            raise ValueError("shared_marginal auxiliary control requires independent actual h release")
        if c.psd_solver not in ("auto", "dense", "tridiagonal"):
            raise ValueError("unknown PSD solver")
        if not isinstance(c.pointwise_fallback, bool):
            raise TypeError("pointwise_fallback must be boolean")
        if c.gate_threshold is not None and (not np.isfinite(c.gate_threshold) or c.gate_threshold <= 0):
            raise ValueError("positive finite gate threshold required")
        if c.pointwise_fallback and c.backend in ("per_scale","structured_independent") and c.gate_threshold is None:
            raise ValueError("per_scale pointwise fallback requires a public, release-specific gate_threshold")
        return c

    def as_dict(self):
        return asdict(self)
