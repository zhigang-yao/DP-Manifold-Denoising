"""Frozen D=10 Fourier-torus geometry generator for the paper simulations.

This module only generates synthetic observations and public query points.
The archived exploratory Fourier PCA pilot is available in Git history.
"""
import numpy as np

RADII = (0.32, 0.32, 0.27, 0.24, 0.15)
FREQUENCIES = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0))
GEOMETRY_VERSION = "fourier_torus_native_D10_r032_032_027_024_015_v1"
SEED = 20261107
D = 10
INTRINSIC_D = 2


def embed(theta):
    """Return positions (N,10) and derivatives (N,10,2)."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 2:
        raise ValueError("angles must be (N,2)")
    r = np.asarray(RADII)
    w = np.asarray(FREQUENCIES, float)
    angles = theta @ w.T
    c, s = np.cos(angles), np.sin(angles)
    points = np.empty((len(theta), D))
    points[:, 0::2] = c * r
    points[:, 1::2] = s * r
    tangent = np.empty((len(theta), D, INTRINSIC_D))
    tangent[:, 0::2, :] = -s[:, :, None] * r[None, :, None] * w[None, :, :]
    tangent[:, 1::2, :] = c[:, :, None] * r[None, :, None] * w[None, :, :]
    return points, tangent


def metric():
    w = np.asarray(FREQUENCIES, float)
    r = np.asarray(RADII, float)
    return (w.T * (r * r)) @ w


def geometry_checks():
    gram = metric()
    covariance_eigenvalues = np.repeat(np.asarray(RADII) ** 2 / 2, 2)
    return dict(geometry_version=GEOMETRY_VERSION, ambient_dimension=D,
                intrinsic_dimension=INTRINSIC_D,
                induced_metric=gram.tolist(),
                area=float((2 * np.pi) ** 2 * np.sqrt(np.linalg.det(gram))),
                covariance_eigenvalues=covariance_eigenvalues.tolist(),
                full_affine_rank=bool(np.all(covariance_eigenvalues > 0)),
                constant_surface_volume_density=True,
                point_norm=float(np.linalg.norm(RADII)))


def _rng(role, replicate=0, n=0):
    return np.random.default_rng(np.random.SeedSequence([SEED, role, replicate, n]))


def sample_unit(n, Q, replicate, *, sigma=0.04, a=0.08):
    """Uniform manifold-area reference and full 8D normal query offsets."""
    if n < 2 or Q < 1 or not (0 <= sigma < 1) or not (0 <= a < 1):
        raise ValueError("invalid configuration")
    from geometry.noise import sample_uniform_ball_noise
    ref_theta = _rng(1, replicate, n).uniform(0, 2 * np.pi, size=(n, 2))
    clean, _ = embed(ref_theta)
    latent_theta = _rng(3, replicate).uniform(0, 2 * np.pi, size=(Q, 2))
    latent, tangent = embed(latent_theta)
    basis, _ = np.linalg.qr(tangent, mode="reduced")
    g = _rng(4, replicate).standard_normal((Q, D))
    orthogonal = g - np.einsum("qdi,qi->qd", basis,
                              np.einsum("qdi,qd->qi", basis, g))
    length = np.linalg.norm(orthogonal, axis=1)
    if np.any(length <= 1.e-10):
        raise ArithmeticError("degenerate normal direction")
    normals = orthogonal / length[:, None]
    queries = latent + a * normals
    noise = sample_uniform_ball_noise(n, D, sigma, _rng(2, replicate, n))
    return dict(reference=clean + noise, clean_reference=clean,
                latent=latent, latent_theta=latent_theta, queries=queries,
                normals=normals, tangent=tangent,
                normal_dot_tangent=np.einsum('qd,qdi->qi', normals, tangent),
                public_query_offset=float(np.max(np.abs(
                    np.linalg.norm(queries - latent, axis=1) - a))))


