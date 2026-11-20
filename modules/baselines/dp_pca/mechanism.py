"""One-release Gaussian global affine PCA with private eigengap rank selection.

The private query is (mean, svec(second moment) / sqrt(2)), after clipping
records to a public zero-centred ball. Its conservative replace-one L2
sensitivity is sqrt(4*c**2 + c**4) / n. Rank selection, centring, and every
prediction use only the one released moment vector and public information.

The moment mechanism preserves the archived DP-PCA baseline. The new rank
rule takes the largest adjacent gap in the descending *private* spectrum,
with public candidate ranks 1 through D-1 and smallest-rank tie breaking.
See docs/DP_PCA_BASELINE.md for derivation, provenance, and privacy scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from modules.privacy import analytic_gaussian_sigma, gaussian_dp_delta


COMMON_CLIP_RADIUS = 1.3


def public_clip_radius(geometry: str, sigma: float) -> float:
    """Public theoretical reference support bound for the six campaign geometries.

    Use the historical common radius 1.3 whenever it covers the clean geometry
    plus uniform-ball noise of radius ``sigma``. Otherwise expand using that
    public support bound. No sampled records, latent points, or query offsets
    enter this calculation. The versioned embeddings preserve Euclidean norm.
    """
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be a finite nonnegative public noise radius")
    clean_bounds = {
        "trefoil": np.sqrt(10.0) / 3.0,
        "torus": 1.0,
        "bowl4": np.sqrt(1.0 + .4**2),
    }
    if geometry == "swiss_roll_candidate":
        from geometry.candidate_swiss import (
            HALF_HEIGHT, RADIAL_SLOPE, RADIUS_START,
            SURFACE_AREA_CENTER, THETA_MAX,
        )
        outer_radius = RADIUS_START + RADIAL_SLOPE * THETA_MAX
        centre_xy = np.linalg.norm(SURFACE_AREA_CENTER[:2])
        clean_bound = np.hypot(outer_radius + centre_xy, HALF_HEIGHT)
    elif geometry == "linked_tori":
        from geometry.linked_tori import CENTER_OFFSET, MAJOR_RADIUS, MINOR_RADIUS
        clean_bound = CENTER_OFFSET + MAJOR_RADIUS + MINOR_RADIUS
    elif geometry == "double_graph":
        from geometry.synthetic import DOUBLE_GRAPH_ALPHA
        clean_bound = np.sqrt(1.0 + DOUBLE_GRAPH_ALPHA**2 + (DOUBLE_GRAPH_ALPHA/2)**2)
    elif geometry in clean_bounds:
        clean_bound = clean_bounds[geometry]
    else:
        raise ValueError("geometry has no audited public PCA clipping bound")
    bound = float(clean_bound + sigma)
    if not np.isfinite(bound):
        raise ValueError("public support radius is not representable")
    if bound <= COMMON_CLIP_RADIUS:
        return COMMON_CLIP_RADIUS
    return float(np.nextafter(bound, np.inf))


def _rank_bounds(dimension: int, rank_min: int, rank_max: int | None) -> tuple[int, int]:
    upper = dimension - 1 if rank_max is None else rank_max
    if (not isinstance(rank_min, Integral) or isinstance(rank_min, (bool, np.bool_))
            or not isinstance(upper, Integral) or isinstance(upper, (bool, np.bool_))
            or not 1 <= rank_min <= upper < dimension):
        raise ValueError("public rank bounds must satisfy 1 <= rank_min <= rank_max < D")
    return int(rank_min), int(upper)


def select_eigengap_rank(eigenvalues, *, rank_min: int = 1, rank_max: int | None = None) -> int:
    """Select a rank from a descending, already-private algebraic spectrum.

    No PSD truncation, absolute-value ordering, relative gaps, or raw-data
    validation enters this rule. An exact tie selects the smallest candidate.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.any(np.diff(values) > 0)):
        raise ValueError("finite descending eigenvalues with D >= 2 required")
    lower, upper = _rank_bounds(len(values), rank_min, rank_max)
    gaps = values[:-1] - values[1:]
    return lower + int(np.argmax(gaps[lower - 1:upper]))


def _reference_matrix(reference) -> np.ndarray:
    data = np.asarray(reference, dtype=np.float64)
    if (data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2
            or not np.isfinite(data).all()):
        raise ValueError("reference must be a finite n-by-D matrix with n >= 2 and D >= 2")
    return data


def _query_matrix(queries, dimension: int) -> np.ndarray:
    data = np.asarray(queries, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != dimension or not np.isfinite(data).all():
        raise ValueError("queries must be a finite Q-by-D public matrix")
    return data


def _clip_rows(reference: np.ndarray, radius: float) -> np.ndarray:
    # Preserve the historical arithmetic for ordinary finite-norm records.
    # A stable fallback also clips finite records whose squared norm overflows.
    with np.errstate(over="ignore"):
        norms = np.linalg.norm(reference, axis=1)
    clipped = reference * np.minimum(1., radius / np.maximum(norms, 1e-300))[:, None]
    overflow = ~np.isfinite(norms)
    if np.any(overflow):
        scaled = reference[overflow] / np.max(np.abs(reference[overflow]), axis=1)[:, None]
        clipped[overflow] = radius * scaled / np.linalg.norm(scaled, axis=1)[:, None]
    return clipped


@dataclass(frozen=True)
class DPPCA:
    """Released moments and their postprocessed affine projection model."""

    mean: np.ndarray
    second_moment: np.ndarray
    eigenvalues: np.ndarray  # descending algebraic eigenvalues, before PSD repair
    components: np.ndarray  # D-by-rank orthonormal basis of the leading eigenspace
    rank: int
    metadata: dict

    def predict(self, queries) -> np.ndarray:
        """Project any public query batch without a new private release."""
        x = _query_matrix(queries, len(self.mean))
        return self.mean + (x - self.mean) @ self.components @ self.components.T


def fit_dp_pca(reference, *, epsilon: float, delta: float, radius: float,
               rng: np.random.Generator, rank_min: int = 1,
               rank_max: int | None = None) -> DPPCA:
    """Fit one (epsilon, delta)-DP affine PCA model under fixed-n replacement.

    ``n``, ``D``, clipping centre zero, ``radius``, and candidate rank bounds
    are public. The supplied generator supports reproducible synthetic
    experiments; its state/seeds must not be disclosed for a real private
    release. Further independent fits on the same private data compose.
    """
    y = _reference_matrix(reference)
    n, dimension = y.shape
    lower, upper_rank = _rank_bounds(dimension, rank_min, rank_max)
    epsilon, delta, radius = float(epsilon), float(delta), float(radius)
    if (not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(delta)
            or not 0 < delta < 1 or not np.isfinite(radius) or radius <= 0):
        raise ValueError("finite epsilon > 0, 0 < delta < 1, and radius > 0 required")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        sensitivity = float(np.sqrt(4. * np.float64(radius)**2 + np.float64(radius)**4) / n)
    if not np.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("radius does not yield a representable positive sensitivity")
    unit_sd, _ = analytic_gaussian_sigma(epsilon, delta, 1.)
    noise_sd = float(unit_sd * sensitivity)
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise ValueError("Gaussian noise scale is not representable")
    achieved_delta = gaussian_dp_delta(epsilon, noise_sd, sensitivity)
    while achieved_delta > delta:
        noise_sd = float(np.nextafter(noise_sd, np.inf))
        achieved_delta = gaussian_dp_delta(epsilon, noise_sd, sensitivity)

    clipped = _clip_rows(y, radius)
    mean = clipped.mean(axis=0)
    second = clipped.T @ clipped / n
    # Both blocks are one isotropic Gaussian draw in the joint feature space.
    mean = mean + noise_sd * rng.standard_normal(dimension)
    upper = np.triu_indices(dimension)
    diagonal_scale = np.where(upper[0] == upper[1], np.sqrt(2.), 1.)
    packed_second = second[upper] / diagonal_scale
    packed_second += noise_sd * rng.standard_normal(len(packed_second))
    released_upper = np.zeros((dimension, dimension), dtype=np.float64)
    released_upper[upper] = packed_second * diagonal_scale
    second = released_upper + np.triu(released_upper, 1).T

    covariance = second - np.outer(mean, mean)
    covariance = .5 * (covariance + covariance.T)
    ascending_values, vectors = np.linalg.eigh(covariance)
    eigenvalues = ascending_values[::-1].copy()
    rank = select_eigengap_rank(eigenvalues, rank_min=lower, rank_max=upper_rank)
    # Preserve the legacy leading-component column order for fixed-rank parity.
    components = vectors[:, -rank:].copy()
    gaps = eigenvalues[:-1] - eigenvalues[1:]
    metadata = {
        "method": "DP-PCA-eigengap",
        "model": "global_affine_pca",
        "adjacency": "fixed_n_replace_one",
        "epsilon": epsilon,
        "delta": delta,
        "achieved_delta": float(achieved_delta),
        "n": int(n),
        "D": int(dimension),
        "clip_radius": radius,
        "clip_center": [0.] * dimension,
        "clip_radius_source": "public_argument",
        "moment_normalization": "1/n",
        "joint_feature": "(mean, svec(second_moment)/sqrt(2))",
        "sensitivity": sensitivity,
        "noise_sd": noise_sd,
        "accountant": "analytic_gaussian_balle_wang_2018",
        "release_count": 1,
        "rank_rule": "largest_adjacent_gap_descending_private_algebraic_spectrum",
        "rank_min": lower,
        "rank_max": upper_rank,
        "rank_tie_break": "smallest_rank",
        "selected_rank": rank,
        "selected_gap": float(gaps[rank - 1]),
        "candidate_ranks": list(range(lower, upper_rank + 1)),
        "candidate_gaps": gaps[lower - 1:upper_rank].tolist(),
        "psd_repair": False,
        "scope": "one_reference_model_and_all_public_query_predictions",
        "moment_provenance": "archive/m20-dp-pca-20261106@a5edc7d68aaf4ec641913055fced27d3d9aee758",
    }
    return DPPCA(mean=mean, second_moment=second, eigenvalues=eigenvalues,
                 components=components, rank=rank, metadata=metadata)


def run_dp_pca(reference, queries, *, epsilon: float, delta: float, radius: float,
               rng: np.random.Generator, rank_min: int = 1,
               rank_max: int | None = None) -> dict:
    """Campaign interface: return estimates, selected rank, spectrum, metadata."""
    y = _reference_matrix(reference)
    x = _query_matrix(queries, y.shape[1])
    model = fit_dp_pca(y, epsilon=epsilon, delta=delta, radius=radius, rng=rng,
                       rank_min=rank_min, rank_max=rank_max)
    return {
        "estimates": model.predict(x),
        "rank": model.rank,
        "eigenvalues": model.eigenvalues,
        "metadata": dict(model.metadata),
    }
