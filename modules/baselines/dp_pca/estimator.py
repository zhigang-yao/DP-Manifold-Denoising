"""Private-clip, single-release DP-PCA: eigengap, nu-eigenratio and paper Soft.

Research comparison. Clipping-radius selection never inspects geometry, latent
targets or a true support bound. All three predictors are postprocessing of
the SAME joint Gaussian moment release. Releasing reproducibility seeds or
unnoised synthetic references is not a private real-data release.
"""
from __future__ import annotations

import numpy as np

from modules.baselines.dp_pca.mechanism import fit_dp_pca

PUBLIC_CLIP_GRID = np.exp2(np.arange(-192, 193, dtype=float) / 16)
DEFAULT_ALPHA = 0.1
DEFAULT_NU_DRAWS = 10


def private_quantile_radius(norms, quantile, epsilon, rng):
    """Exponential mechanism over a geometry-independent public log grid.

    Utility is -abs(count(norm<=c) - q*N). Under fixed-size replace-one
    adjacency its sensitivity is at most one, for every public candidate c.
    """
    norms = np.asarray(norms, dtype=float)
    if norms.ndim != 1 or not len(norms) or not np.isfinite(norms).all() or (norms < 0).any():
        raise ValueError("norms must be a nonempty finite nonnegative vector")
    if not (0 < quantile < 1 and np.isfinite(epsilon) and epsilon > 0):
        raise ValueError("invalid quantile / epsilon")
    counts = np.searchsorted(np.sort(norms), PUBLIC_CLIP_GRID, side="right")
    utility = -np.abs(counts - quantile * len(norms))
    logits = 0.5 * epsilon * utility
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return float(rng.choice(PUBLIC_CLIP_GRID, p=probabilities))


def covariance_noise_rms(released_mean, noise_sd, rng, draws=DEFAULT_NU_DRAWS):
    """First-order RMS spectral perturbation using public auxiliary noise.

    L = W - mu_tilde g^T - g mu_tilde^T. The true uncentered-minus-centered
    difference also has a second-order -gg^T term, omitted by design.
    This nu is not a finite-sample high-probability operator-norm bound.
    """
    mu = np.asarray(released_mean, dtype=float)
    D = len(mu)
    if D < 2 or not np.isfinite(mu).all() or noise_sd <= 0 or draws < 1:
        raise ValueError("invalid mean, noise scale or auxiliary draw count")
    upper = np.triu_indices(D)
    diag = np.where(upper[0] == upper[1], np.sqrt(2.0), 1.0)
    squares = []
    for _ in range(draws):
        g = noise_sd * rng.standard_normal(D)
        W = np.zeros((D, D), dtype=float)
        W[upper] = noise_sd * rng.standard_normal(len(upper[0])) * diag
        W += np.triu(W, 1).T
        L = W - np.outer(mu, g) - np.outer(g, mu)
        squares.append(float(np.max(np.abs(np.linalg.eigvalsh(L))) ** 2))
    return float(np.sqrt(np.mean(squares)))


def spectral_rules(eigenvalues, nu, *, alpha=DEFAULT_ALPHA):
    """One private descending algebraic spectrum, three explicit methods."""
    vals = np.asarray(eigenvalues, dtype=float)
    if vals.ndim != 1 or len(vals) < 2 or not np.isfinite(vals).all():
        raise ValueError("finite eigenvalue vector of length >=2 required")
    if np.any(np.diff(vals) > 1e-12) or not np.isfinite(nu) or nu < 0:
        raise ValueError("descending spectrum and nonnegative nu required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    positive = np.maximum(vals, 0)
    b = np.maximum(positive, max(float(nu), 1e-12))
    rank_gap = 1 + int(np.argmax(vals[:-1] - vals[1:]))
    rank_ratio = 1 + int(np.argmax(b[:-1] / b[1:]))
    threshold = float(max(np.sqrt(b[0] * b[-1]), alpha * b[0]))
    eligible = b[b <= threshold * (1.0 + 1e-13)]
    if eligible.size == 0:
        raise ArithmeticError("unexpected empty paper-soft noise group")
    b_star = float(eligible.max())
    lam = float(np.sqrt(b[0] * b_star))
    weights = (positive**2 / (positive**2 + lam**2)
               if lam > 0 else (positive > 0).astype(float))
    return {
        "rank_gap": rank_gap,
        "rank_ratio": rank_ratio,
        "soft_lambda": lam,
        "soft_threshold": threshold,
        "soft_b_star": b_star,
        "soft_weights": weights,
    }


def project_hard(queries, mean, eigenvectors_desc, rank):
    U = eigenvectors_desc[:, :rank]
    return mean + (queries - mean) @ U @ U.T


def project_soft(queries, mean, eigenvectors_desc, weights):
    return mean + ((queries - mean) @ eigenvectors_desc * weights) @ eigenvectors_desc.T


def release_and_compare(reference, queries, *, epsilon, delta, quantile,
                        privacy_mode, rng_clip, rng_pca, rng_aux,
                        nu_draws=DEFAULT_NU_DRAWS, alpha=DEFAULT_ALPHA):
    """Private radius then one Gaussian PCA fit, all three paired outputs.

    split: deterministic first 10% select radius with (eps,0)-DP; the
           remaining 90% fit PCA with (eps,delta)-DP. Disjoint fixed blocks.
    compose: all records select with .15eps, then fit with .85eps.
    Total privacy in either case <= (eps,delta), for replace-one records.
    """
    X = np.asarray(reference, dtype=float)
    Z = np.asarray(queries, dtype=float)
    if X.ndim != 2 or Z.ndim != 2 or X.shape[1] != Z.shape[1] or len(X) < 20:
        raise ValueError("expected reference n-by-D and public queries m-by-D")
    n = len(X)
    if privacy_mode == "split":
        n_clip = int(0.1 * n)
        selection, training = X[:n_clip], X[n_clip:]
        epsilon_clip = epsilon_pca = epsilon
    elif privacy_mode == "compose":
        selection = training = X
        epsilon_clip, epsilon_pca = 0.15 * epsilon, 0.85 * epsilon
    else:
        raise ValueError("privacy_mode must be split or compose")
    radius = private_quantile_radius(
        np.linalg.norm(selection, axis=1), quantile, epsilon_clip, rng_clip)
    model = fit_dp_pca(training, epsilon=epsilon_pca, delta=delta,
                       radius=radius, rng=rng_pca)
    covariance = model.second_moment - np.outer(model.mean, model.mean)
    covariance = 0.5 * (covariance + covariance.T)
    values_asc, vectors_asc = np.linalg.eigh(covariance)
    values = values_asc[::-1].copy()
    vectors = vectors_asc[:, ::-1].copy()
    nu = covariance_noise_rms(model.mean, model.metadata["noise_sd"],
                              rng_aux, draws=nu_draws)
    rules = spectral_rules(values, nu, alpha=alpha)
    predictions = {
        "gap": project_hard(Z, model.mean, vectors, rules["rank_gap"]),
        "ratio": project_hard(Z, model.mean, vectors, rules["rank_ratio"]),
        "soft": project_soft(Z, model.mean, vectors, rules["soft_weights"]),
    }
    return {
        "predictions": predictions,
        "radius": radius,
        "epsilon_clip": epsilon_clip,
        "epsilon_pca": epsilon_pca,
        "n_select": len(selection),
        "n_fit": len(training),
        "nu": nu,
        "noise_sd": model.metadata["noise_sd"],
        "private_spectrum": values,
        **rules,
        "release_count": 1,
    }
