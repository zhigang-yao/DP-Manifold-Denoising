"""Frozen analytic densities, independent streams, Stage13 normal queries."""

import numpy as np
from scipy.special import i0e, i1e
from . import manifolds as g
from geometry.queries import make_query_bank
from geometry.noise import sample_uniform_ball_noise


def project(case, x):
    x = np.asarray(x)
    if case == "affine":
        y = np.zeros_like(x)
        y[:, 0] = np.clip(x[:, 0], -1, 1)
        return dict(
            points=y,
            boundary_distance=1 - np.abs(y[:, 0]),
            objective_gap_upper=np.zeros(len(x)),
        )
    return g.project_with_audit(case, x)


def density_weight(case, points, regime):
    if regime == "bimodal":
        if case != "circle":
            raise ValueError("bimodal density is frozen for Circle")
        theta = np.arctan2(points[:, 1], points[:, 0])
        return 1 + 0.8 * np.cos(2 * theta)
    if regime == "valley":
        if case != "torus":
            raise ValueError("density valley is frozen for Torus")
        # Relative to uniform surface volume, whose minor-angle density is
        # (1 + .5*cos(phi))/(2*pi). i0e/i1e include exp(-kappa).
        meta = g.geometry_metadata(case)
        raw = points / meta["scale"] + np.asarray(meta["raw_center"])
        theta = np.arctan2(raw[:, 1], raw[:, 0])
        phi = np.arctan2(raw[:, 2], np.hypot(raw[:, 0], raw[:, 1]) - 2 / 3)
        bump = np.exp(2 * (np.cos(theta) + np.cos(phi) - 2))
        normalizer = 1 - 0.9 * i0e(2) * (i0e(2) + 0.5 * i1e(2))
        return (1 - 0.9 * bump) / normalizer
    # Density w.r.t. normalized manifold volume. S-curve symmetry gives E[x1]=0,
    # so 1+beta*x1/(scale) integrates to 1; support x1/scale in [-1,1].
    beta = {"uniform": 0.0, "moderate": 0.5, "strong": 0.9}[regime]
    if beta and case != "s_curve":
        raise ValueError("density study frozen for S-curve")
    scale = g.geometry_metadata(case)["scale"] if case != "affine" else 1.0
    return 1 + beta * points[:, 0] / scale


def density_envelope(regime):
    """Analytic rejection envelope relative to normalized volume."""
    if regime == "valley":
        return 1 / (1 - 0.9 * i0e(2) * (i0e(2) + 0.5 * i1e(2)))
    return {"uniform": 1.0, "moderate": 1.5, "strong": 1.9, "bimodal": 1.8}[regime]


def density_regions(case, points, regime):
    """Prespecified spatial strata, unchanged between methods/replicates."""
    if case == "s_curve":
        coordinate = points[:, 0] / g.geometry_metadata(case)["scale"]
        return np.where(
            coordinate < -1 / 3, "low", np.where(coordinate > 1 / 3, "high", "middle")
        )
    # Use the same spatial masks for a nonuniform scenario and its uniform
    # control; labels mean reference density only in the nonuniform scenario.
    if case == "circle":
        value = density_weight(case, points, "bimodal")
        return np.where(value < 0.6, "low", np.where(value > 1.4, "high", "middle"))
    if case == "torus":
        normalizer = 1 / density_envelope("valley")
        bump = (1 - density_weight(case, points, "valley") * normalizer) / 0.9
        return np.where(bump >= 0.4, "low", np.where(bump >= 0.1, "middle", "high"))
    return np.full(len(points), "all")


def generate(c):
    if (
        c.geometry not in ("affine", "circle", "torus", "s_curve", "sphere3")
        or c.n < 2
        or not 1 <= c.m <= 100
        or c.sigma < 0
        or c.a < 0
    ):
        raise ValueError("invalid frozen design")
    index = ("affine", "circle", "torus", "s_curve", "sphere3").index(c.geometry)
    # Same reference and query identities across epsilon/noise-radius scans.
    ref_rng = np.random.default_rng(
        np.random.SeedSequence([20261022, c.replicate, index, 1])
    )
    noise_rng = np.random.default_rng(
        np.random.SeedSequence([20261022, c.replicate, index, 2])
    )
    if c.geometry == "affine":
        D, d = 3, 1
        clean = np.c_[ref_rng.uniform(-1, 1, c.n), np.zeros((c.n, 2))]
        qr = np.random.default_rng(np.random.SeedSequence([20261022, index, 3]))
        latent = np.c_[qr.uniform(-0.89, 0.89, c.m), np.zeros((c.m, 2))]
        direction = np.c_[np.zeros(c.m), qr.normal(size=(c.m, 2))]
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        queries = latent + c.a * direction
        boundary = 1 - np.abs(latent[:, 0])
    else:
        meta = g.geometry_metadata(c.geometry)
        D = meta["native_dimension"]
        d = meta["d"]
        bank = make_query_bank(c.geometry, c.m, 20261022, [c.a])
        latent = bank["base"]
        queries = bank["points"][c.a]
        direction = bank["directions"]
        boundary = bank["base_boundary_distance"]
        chunks = []
        left = c.n
        for _ in range(10000):
            candidate = (
                g._sample_raw(c.geometry, left, ref_rng)
                - np.asarray(meta["raw_center"])
            ) * meta["scale"]
            if c.density != "uniform":
                w = density_weight(c.geometry, candidate, c.density)
                candidate = candidate[ref_rng.random(len(candidate)) < w / density_envelope(c.density)]
            chunks.append(candidate)
            left -= len(candidate)
            if left == 0:
                break
        else:
            raise RuntimeError("density rejection sampler incomplete")
        clean = np.concatenate(chunks)
    if c.noise == "bounded":
        noise = sample_uniform_ball_noise(c.n, D, c.sigma, noise_rng)
    elif c.noise == "gaussian":
        noise = noise_rng.normal(size=(c.n, D)) * c.sigma / np.sqrt(D + 2)
    else:
        raise ValueError("unknown reference noise")
    audit = project(c.geometry, queries)
    if np.max(np.linalg.norm(audit["points"] - latent, axis=1)) > 3e-7:
        raise ValueError("nearest projection audit failed")
    if c.geometry in ("affine", "s_curve") and np.min(boundary) <= 0.1:
        raise ValueError("interior query audit failed")
    return dict(
        reference=clean + noise,
        clean_reference=clean,
        queries=queries,
        latent=latent,
        projection=audit["points"],
        directions=direction,
        boundary_distance=boundary,
        query_density=density_weight(c.geometry, latent, c.density),
        d=d,
        D=D,
    )
