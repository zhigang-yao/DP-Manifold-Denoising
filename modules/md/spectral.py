"""Canonical eigenpair ordering from the validated spectral ablation."""

import numpy as np


def canonical_eigenpairs(C):
    values, vectors = np.linalg.eigh((C + C.swapaxes(-1, -2)) * 0.5)
    D = C.shape[-1]
    vv = vectors.reshape(-1, D, D)
    ss = values.reshape(-1, D)
    for i in np.flatnonzero(np.any(np.diff(ss, axis=1) == 0, axis=1)):
        s = ss[i]
        U = vv[i]
        begin = 0
        while begin < D:
            end = begin + 1
            while end < D and s[end] == s[begin]:
                end += 1
            if end - begin > 1:
                P = U[:, begin:end] @ U[:, begin:end].T
                basis = []
                for axis in range(D):
                    v = P[:, axis].copy()
                    for _ in range(2):
                        for b in basis:
                            v -= b * np.dot(b, v)
                    norm = np.linalg.norm(v)
                    if norm > 64 * np.finfo(float).eps * D:
                        basis.append(v / norm)
                    if len(basis) == end - begin:
                        break
                if len(basis) != end - begin:
                    raise ArithmeticError("canonical basis failed")
                U[:, begin:end] = np.column_stack(basis)
            begin = end
    return values, vectors


from modules.md.moments import unpack
from collections.abc import Mapping

CLIPPED_SOFT_LIPSCHITZ = 0.824727084


def _array(value, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite numeric array")
    return result


def _public_array(value, shape, name, *, positive=False):
    result = _array(value, name)
    try:
        result = np.broadcast_to(result, shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to moment shape {shape}") from exc
    if np.any(result <= 0 if positive else result < 0):
        adjective = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {adjective}")
    return result


def _public_settings(shape, h, sd, coeff):
    if not isinstance(coeff, Mapping):
        raise ValueError("coeff must contain public c0, cv, and cb coefficients")
    try:
        cb = coeff["cb"] if "cb" in coeff else coeff["cB"]
        c0, cv = coeff["c0"], coeff["cv"]
    except KeyError as exc:
        raise ValueError(
            "coeff must contain public c0, cv, and cb coefficients"
        ) from exc
    return (
        _public_array(h, shape, "h", positive=True),
        _public_array(sd, shape, "sd", positive=True),
        _public_array(c0, shape, "c0"),
        _public_array(cv, shape, "cv"),
        _public_array(cb, shape, "cb", positive=True),
    )


def _moments(q, vector, matrix):
    q = _array(q, "mass")
    vector = _array(vector, "first moment")
    matrix = _array(matrix, "second moment")
    if vector.ndim != q.ndim + 1 or vector.shape[:-1] != q.shape:
        raise ValueError("first-moment shape must be mass.shape + (D,)")
    D = vector.shape[-1]
    if D < 1 or matrix.shape != q.shape + (D, D):
        raise ValueError("second-moment shape must be mass.shape + (D,D)")
    return q, vector, 0.5 * matrix + 0.5 * matrix.swapaxes(-1, -2)


def _proxy(q, mean_norm, second_norm, h, sd, c0, cv, cb):
    ratio = sd / q
    u = ratio * (h * cv + c0 * mean_norm)
    nu = ratio * (h * h * cb + c0 * second_norm) + 2 * mean_norm * u + 3 * u * u
    return u, nu


def normalize_and_proxy(q, v, B, h, sd, coeff):
    """Normalize natural raw Gaussian moments and recompute the actual proxy.

    Nonpositive mass has valid=False and zero placeholder means/second/nu.
    No empirical gate is applied. Positive-mass rows use denominator q itself.
    Numerical overflow is an explicit error, rather than a hidden RAW return.
    """
    q, v, B = _moments(q, v, B)
    h, sd, c0, cv, cb = _public_settings(q.shape, h, sd, coeff)
    valid = q > 0
    denominator = np.where(valid, q, 1.0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        means = (
            h[..., None] * np.where(valid[..., None], v, 0.0) / denominator[..., None]
        )
        second = (
            h[..., None, None] ** 2
            * np.where(valid[..., None, None], B, 0.0)
            / denominator[..., None, None]
        )
        mean_norm = np.linalg.norm(means, axis=-1)
        second_norm = np.max(np.abs(np.linalg.eigvalsh(second)), axis=-1)
        u, nu = _proxy(denominator, mean_norm, second_norm, h, sd, c0, cv, cb)
    if np.any(valid & (nu <= 0)):
        raise FloatingPointError(
            "positive proxy underflowed outside the supported numerical range"
        )
    return dict(
        mass=q,
        means=means,
        second=second,
        nu=np.where(valid, nu, 0.0),
        valid=valid,
        mean_proxy=np.where(valid, u, 0.0),
        noise_sd=sd,
        noise_coefficients={"c0": c0, "cv": cv, "cb": cb},
        proxy_kind="nominal_expected_scale",
    )


def decode_global_soft(queries, transcript):
    """Call the unchanged legacy Global decoder with multiplier exactly one.

    Any leading batch shape is flattened only for the frozen correction.
    The paper interface supports ambient D>=2.
    """
    q, means, second = _moments(
        transcript["mass"], transcript["means"], transcript["second"]
    )
    D = means.shape[-1]
    if D < 2:
        raise ValueError("the legacy exp65 decoder requires ambient D>=2")
    queries = _array(queries, "queries")
    try:
        queries = np.broadcast_to(queries, means.shape)
        nu = np.broadcast_to(_array(transcript["nu"], "nu"), q.shape)
        valid = np.broadcast_to(np.asarray(transcript["valid"], dtype=bool), q.shape)
    except ValueError as exc:
        raise ValueError(
            "queries, nu, and valid must match the transcript batch shape"
        ) from exc
    if np.any(nu < 0):
        raise ValueError("nu must be nonnegative")
    result = decode_soft(
        queries.reshape(-1, D),
        means.reshape(-1, D),
        second.reshape(-1, D, D),
        nu.reshape(-1),
        rule="global",
        multiplier=1.0,
        valid=valid.reshape(-1),
    )
    return {
        key: np.asarray(value).reshape(q.shape + np.asarray(value).shape[1:])
        for key, value in result.items()
    }


def _lambda_intervals(values, nu, nu_lower, nu_upper, eta):
    """All spectra are descending signed eigenvalues before scalar clipping."""
    positive = np.maximum(values, 0.0)
    lower = np.maximum(values - eta[..., None], 0.0)
    upper = np.maximum(values + eta[..., None], 0.0)
    diagnostics = {}
    lam = np.sqrt(np.maximum(positive[..., 0], nu) * nu)
    lam_lower = np.sqrt(np.maximum(lower[..., 0], nu_lower) * nu_lower)
    lam_upper = np.sqrt(np.maximum(upper[..., 0], nu_upper) * nu_upper)
    return lam, lam_lower, lam_upper, diagnostics


def global_soft_radius(
    mass,
    means,
    second,
    sd,
    h,
    coeff,
    tq,
    tv,
    tb,
    *,
    raw_bounds_kind="conditional",
):
    """Bound private soft versus the same soft rule on any allowed raw triple.

    tq is a scalar raw-mass radius. tv/tb bound physical raw h*v / h^2*B
    errors in Euclidean/operator norm. All radii must be jointly valid for a
    probabilistic claim or data-dependent bandwidth selection. nu is always
    recomputed from the supplied mass/means/second and public settings.

    The maintained Global rule uses no intrinsic-dimension argument.
    """
    if raw_bounds_kind not in ("conditional", "nominal"):
        raise ValueError("raw_bounds_kind must be 'conditional' or 'nominal'")
    mass, means, second = _moments(mass, means, second)
    h, sd, c0, cv, cb = _public_settings(mass.shape, h, sd, coeff)
    tq = _public_array(tq, mass.shape, "tq")
    tv = _public_array(tv, mass.shape, "tv")
    tb = _public_array(tb, mass.shape, "tb")
    valid = mass > tq
    positive_mass = mass > 0
    mass_lower, mass_upper = mass - tq, mass + tq
    safe_mass = np.where(positive_mass, mass, 1.0)
    safe_lower = np.where(valid, mass_lower, 1.0)
    safe_upper = np.where(valid, mass_upper, 1.0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean_norm = np.linalg.norm(means, axis=-1)
        second_norm = np.max(np.abs(np.linalg.eigvalsh(second)), axis=-1)
        mean_error = (tv + mean_norm * tq) / safe_lower
        second_error = (tb + second_norm * tq) / safe_lower
        covariance_error = second_error + 2 * mean_norm * mean_error + mean_error**2
        covariance = second - means[..., :, None] * means[..., None, :]
        values = np.linalg.eigvalsh(
            0.5 * covariance + 0.5 * covariance.swapaxes(-1, -2)
        )[..., ::-1]
        _, nu = _proxy(safe_mass, mean_norm, second_norm, h, sd, c0, cv, cb)
        _, nu_lower = _proxy(
            safe_upper,
            np.maximum(mean_norm - mean_error, 0.0),
            np.maximum(second_norm - second_error, 0.0),
            h,
            sd,
            c0,
            cv,
            cb,
        )
        _, nu_upper = _proxy(
            safe_lower,
            mean_norm + mean_error,
            second_norm + second_error,
            h,
            sd,
            c0,
            cv,
            cb,
        )
        lam, lam_lower, lam_upper, diagnostics = _lambda_intervals(
            values, nu, nu_lower, nu_upper, covariance_error
        )
        if np.any(positive_mass & (lam <= 0)):
            raise FloatingPointError(
                "positive lambda underflowed outside the supported numerical range"
            )
        param_error = np.maximum(
            np.abs(lam - lam_lower) / (lam + lam_lower),
            np.abs(lam_upper - lam) / (lam_upper + lam),
        )
        operator_error = np.minimum(
            1.0, CLIPPED_SOFT_LIPSCHITZ * covariance_error / lam + param_error
        )
        radius = mean_error + mean_norm * operator_error
    result = dict(
        valid=valid,
        R=np.where(valid, radius, np.inf),
        meanerror=np.where(valid, mean_error, np.inf),
        seconderror=np.where(valid, second_error, np.inf),
        coverror=np.where(valid, covariance_error, np.inf),
        mass_lower=mass_lower,
        mass_upper=mass_upper,
        nu=np.where(positive_mass, nu, 0.0),
        nu_lower=np.where(valid, nu_lower, np.nan),
        nu_upper=np.where(valid, nu_upper, np.nan),
        lambda_=np.where(positive_mass, lam, 0.0),
        lambda_lower=np.where(valid, lam_lower, np.nan),
        lambda_upper=np.where(valid, lam_upper, np.nan),
        param_error=np.where(valid, param_error, np.nan),
        operator_error=np.where(valid, operator_error, np.nan),
        signed_eigenvalues=values,
        raw_bounds_kind=raw_bounds_kind,
        probability_guarantee="requires_external_joint_raw_certificate",
        rule="global",
    )
    result.update(diagnostics)
    return result


def quadratic_weights(values, nu):
    """Frozen Global lambda and quadratic-soft weights, from signed eigenvalues."""
    positive = np.maximum(values, 0.0)
    lam = np.sqrt(np.maximum(positive[:, -1], nu) * nu)
    denominator = positive**2 + lam[:, None] ** 2
    weights = np.divide(
        lam[:, None] ** 2, denominator, out=np.ones_like(values), where=denominator > 0
    )
    return lam, weights


def decode_soft(
    queries, means, second, nu, *, rule="global", multiplier=1.0, valid=None,
    eigenpairs=None,
):
    """Frozen quadratic Global filter; no rank or truth enters this rule."""
    if rule != "global" or multiplier != 1.0:
        raise ValueError("paper decoder uses Global multiplier one")
    if eigenpairs is None:
        covariance = second - np.einsum("mi,mj->mij", means, means)
        values, vectors = np.linalg.eigh((covariance + covariance.transpose(0, 2, 1)) * 0.5)
    else:
        values, vectors = map(np.asarray, eigenpairs)
        if values.shape != means.shape or vectors.shape != second.shape:
            raise ValueError("cached eigenpairs must match the moment batch")
        if not np.isfinite(values).all() or not np.isfinite(vectors).all():
            raise ValueError("nonfinite cached eigenpairs")
    nu = np.asarray(nu)
    if np.any(nu < 0) or not np.all(np.isfinite(nu)):
        raise ValueError("nu must be finite and nonnegative")
    lam, weights = quadratic_weights(values, nu)
    coeff = np.einsum("mji,mj->mi", vectors, means)
    estimate = queries + np.einsum("mij,mj->mi", vectors, coeff * weights)
    if valid is not None:
        estimate = np.where(valid[:, None], estimate, queries)
    return dict(estimate=estimate, lambda_=lam)


def hard_rank_correction(queries, moments, h, usable, intrinsic_dimension):
    """Known-d filter only, on the identical stabilized moments and mass gate."""
    q, v, B = unpack(np.asarray(moments), queries.shape[-1])
    D = queries.shape[-1]
    if (
        isinstance(intrinsic_dimension, bool)
        or not isinstance(intrinsic_dimension, (int, np.integer))
        or not 0 < intrinsic_dimension < D
    ):
        raise ValueError("hard rank-d requires integer 0 < d < D")
    safe = np.where(q > 0, q, 1.0)
    mean = h * v / safe[..., None]
    C = h * h * B / safe[..., None, None] - mean[..., :, None] * mean[..., None, :]
    eigenvalues, U = canonical_eigenpairs(C)
    N = U[..., :, : D - intrinsic_dimension]
    correction = np.einsum("...ij,...kj,...k->...i", N, N, mean)
    y = queries + correction
    return np.where(usable[..., None], y, queries), eigenvalues
