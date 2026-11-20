"""Canonical mixture spectrum and public single-query gate/proxy coefficients.

Promoted from the frozen paper vendor source. Preparation and numerical audits
are unchanged; unused alternate noise-sampling and moment-release APIs were
removed. The paper's only joint sampler is modules.md.gaussian_field.FullField.
The ideal covariance statements are distinct from machine-level DP certification.
Natural coordinates are (q,v,svec(B)/sqrt(2)); no private reference enters here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral

import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc, gammaincc, gammaln, hyperu, logsumexp, roots_legendre



_BLOCKS = ("scalar_trace", "vector", "traceless")


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _bandwidths(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("bandwidths must be a nonempty positive finite vector")
    if np.any(np.diff(values) <= 0):
        raise ValueError("bandwidths must be strictly increasing; reuse duplicate readouts")
    return values.copy()


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _spectrum_parameters(dimension, bandwidths, h_min, h_max):
    dimension = _positive_integer(dimension, "dimension")
    hs = _bandwidths(bandwidths)
    lower = _scalar(h_min, "h_min", positive=True)
    upper = _scalar(h_max, "h_max", positive=True)
    if lower > upper or hs[0] < lower or hs[-1] > upper:
        raise ValueError("all bandwidths must lie in the fixed public interval")
    ratio = upper / lower
    log_range = math.log(ratio)
    beta = (hs / lower) ** 2
    if not np.isfinite(beta).all() or not math.isfinite(ratio) or dimension * log_range > 500:
        raise ValueError("public bandwidth design exceeds the numerical range")
    shape = dimension / 2
    c = 2 * math.exp(shape * math.log(shape) - shape - float(gammaln(shape)))
    endpoint = (1 - c / dimension) * (-math.expm1(-dimension * log_range))
    normalizer = 1 + c * log_range + endpoint
    return hs, dict(D=dimension, k=shape, lower=lower, upper=upper,
                    ratio=ratio, L=log_range, beta=beta, c=c, b=endpoint, Z=normalizer)


def _relative_spectrum_log(values, parameters):
    """log(Z*p/p_hmin), by the analytic incomplete-gamma mixture identity."""
    u = np.asarray(values, dtype=float)
    if np.any(u < 0) or not np.isfinite(u).all():
        raise ValueError("radial spectrum arguments must be finite and nonnegative")
    if parameters["L"] == 0:
        return np.zeros_like(u)
    shape, ratio = parameters["k"], parameters["ratio"]
    dimension, log_range = parameters["D"], parameters["L"]
    flat = u.reshape(-1)
    result = np.empty_like(flat)
    zero = flat == 0
    if np.any(zero):
        log_integral = math.log(math.expm1(dimension * log_range) / dimension)
        result[zero] = logsumexp([0., math.log(parameters["b"]) + dimension * log_range,
                                 math.log(parameters["c"]) + log_integral])
    active = ~zero
    x = flat[active]
    if len(x):
        limit = ratio * ratio * x
        lower_difference = gammainc(shape, limit) - gammainc(shape, x)
        upper_difference = gammaincc(shape, x) - gammaincc(shape, limit)
        difference = np.where(x < shape, lower_difference, upper_difference)
        log_integral = np.empty_like(x)
        ordinary = (difference > 0) & (x < 500)
        log_integral[ordinary] = (x[ordinary] + float(gammaln(shape)) - math.log(2)
                                  - shape * np.log(x[ordinary]) + np.log(difference[ordinary]))
        if np.any(~ordinary):
            large = x[~ordinary]
            endpoint_factor = np.exp(dimension * log_range - (ratio * ratio - 1) * large)
            integral = .5 * (hyperu(1., 1 + shape, large)
                             - endpoint_factor * hyperu(1., 1 + shape, ratio * ratio * large))
            if np.any(integral <= 0) or not np.isfinite(integral).all():
                raise ArithmeticError("analytic relative-spectrum evaluation lost positivity")
            log_integral[~ordinary] = np.log(integral)
        endpoint = (math.log(parameters["b"]) + dimension * log_range
                    - (ratio * ratio - 1) * x)
        result[active] = np.logaddexp(np.logaddexp(0., endpoint),
                                     math.log(parameters["c"]) + log_integral)
    return result.reshape(u.shape)


def _relative_spectrum_log_legendre(values, parameters, order):
    """Independent positive Gauss-Legendre evaluation of the log mixture."""
    u = np.asarray(values, dtype=float)
    if parameters["L"] == 0:
        return np.zeros_like(u)
    nodes, weights = roots_legendre(order)
    t = .5 * parameters["L"] * (nodes + 1)
    weights = .5 * parameters["L"] * weights
    terms = (np.log(parameters["c"] * weights) + parameters["D"] * t
             - u[..., None] * np.expm1(2 * t))
    endpoint = (math.log(parameters["b"]) + parameters["D"] * parameters["L"]
                - (parameters["ratio"] ** 2 - 1) * u)
    return np.logaddexp(np.logaddexp(0., endpoint), logsumexp(terms, axis=-1))


def _moments_to_blocks(moments, parameters):
    """moments[s] is h_min**(2*s) times the unnormalized radial integral."""
    dimension, beta = parameters["D"], parameters["beta"]
    a, first, second = moments
    u = first / dimension
    h = beta[:, None] * beta[None, :] * second / (dimension * (dimension + 2))
    cross = math.sqrt(dimension / 2) * (a - u * beta[None, :])
    trace = (dimension * a / 2 - (beta[:, None] + beta[None, :]) * first / 2
             + beta[:, None] * beta[None, :] * second / (2 * dimension))
    return dict(scalar_trace=np.block([[a, cross], [cross.T, trace]]),
                vector=np.sqrt(beta[:, None] * beta[None, :]) * u, traceless=h)


def _gamma_interval_mass(shape, left, right):
    # Select the tail that avoids subtraction of two values near one.
    lower = gammainc(shape, right) - gammainc(shape, left)
    upper = gammaincc(shape, left) - gammaincc(shape, right)
    value = np.where(left < shape, lower, upper)
    if np.any(value < -8 * np.finfo(float).eps):
        raise ArithmeticError("incomplete-gamma interval lost nonnegativity")
    return np.maximum(value, 0.)


def _shell_moments(parameters, bins, tail_u):
    """Exact-real target: a one-sided lower relative spectrum on radial shells."""
    count, shape, beta = len(parameters["beta"]), parameters["k"], parameters["beta"]
    edges = np.r_[0., np.geomspace(1e-8, tail_u, bins)]
    log_m = _relative_spectrum_log(edges, parameters)
    # m_rel is analytically decreasing. Reject material numerical violations.
    if np.max(np.diff(log_m)) > 2e-12:
        raise ArithmeticError("relative spectrum numerical monotonicity check failed")
    inverse_right = np.exp(-log_m[1:])
    log_inflation = max(float(np.max(log_m[:-1] - log_m[1:])), float(log_m[-1]), 0.)
    inflation = float(np.nextafter(math.exp(log_inflation), np.inf))
    moments = np.empty((3, count, count))
    for j in range(count):
        for index in range(j + 1):
            rate = beta[j] + beta[index] - 1
            left, right = rate * edges[:-1], rate * edges[1:]
            for power in range(3):
                exponent = shape + power
                mass = _gamma_interval_mass(exponent, left, right)
                integral = float(np.dot(inverse_right, mass) + gammaincc(exponent, rate * tail_u))
                prefactor = math.exp(math.log(parameters["Z"])
                                     + shape * (math.log(beta[j]) + math.log(beta[index]))
                                     + power * math.log(2) + float(gammaln(exponent) - gammaln(shape))
                                     - exponent * math.log(rate))
                moments[power, j, index] = moments[power, index, j] = prefactor * integral
    return moments, dict(radial_shell_bins=bins, radial_tail_u=tail_u,
                         spectral_inflation_bound=inflation,
                         first_shell_right=1e-8, relative_spectrum_log_endpoints=log_m,
                         radial_shell_edges=edges)


def _original_gram(parameters, order, mixture_order, tail_u):
    """Independent positive-weight quadrature of the original continuous p."""
    dimension, shape, beta = parameters["D"], parameters["k"], parameters["beta"]
    stop = math.sqrt(2 * tail_u)
    edges = [0., min(1., 1 / parameters["ratio"])]
    while edges[-1] < stop:
        edges.append(min(stop, 2 * edges[-1]))
    nodes, weights = roots_legendre(order)
    rho = np.concatenate([.5 * (b - a) * nodes + .5 * (a + b) for a, b in zip(edges[:-1], edges[1:])])
    quadrature_weights = np.concatenate([.5 * (b - a) * weights for a, b in zip(edges[:-1], edges[1:])])
    log_weight = (np.log(quadrature_weights) + math.log(parameters["Z"])
                  + (dimension - 1) * np.log(rho) - rho * rho / 2
                  - (shape - 1) * math.log(2) - float(gammaln(shape))
                  - _relative_spectrum_log_legendre(rho * rho / 2, parameters, mixture_order))
    log_phi = shape * np.log(beta)[:, None] - (beta[:, None] - 1) * rho[None, :] ** 2 / 2
    q = np.exp(log_phi + log_weight[None, :] / 2)
    trace = (dimension - beta[:, None] * rho[None, :] ** 2) * q / math.sqrt(2 * dimension)
    qtrace = np.concatenate((q, trace), axis=0)
    vector = q * rho[None, :] * np.sqrt(beta[:, None] / dimension)
    traceless = q * rho[None, :] ** 2 * beta[:, None] / math.sqrt(dimension * (dimension + 2))
    blocks = dict(scalar_trace=qtrace @ qtrace.T, vector=vector @ vector.T,
                  traceless=traceless @ traceless.T)
    tails = [math.exp(power * math.log(2) + float(gammaln(shape + power) - gammaln(shape)))
             * float(gammaincc(shape + power, tail_u)) for power in range(3)]
    count, z = len(beta), parameters["Z"]
    bounds = dict(scalar_trace=count * z * ((1 + dimension / 2) * tails[0] + tails[1] + tails[2] / (2 * dimension)),
                  vector=count * z * tails[1] / dimension,
                  traceless=count * z * tails[2] / (dimension * (dimension + 2)))
    return blocks, bounds, len(rho)


def _original_adaptive(parameters):
    """All original radial pairs, independent integration and spectrum formula."""
    beta, shape = parameters["beta"], parameters["k"]
    count = len(beta)
    moments, errors = np.empty((3, count, count)), np.empty((3, count, count))
    for j in range(count):
        for index in range(j + 1):
            rate = beta[j] + beta[index] - 1
            for power in range(3):
                exponent = shape + power
                log_prefactor = (math.log(parameters["Z"])
                                  + shape * (math.log(beta[j]) + math.log(beta[index]))
                                  + power * math.log(2) - float(gammaln(shape))
                                  - exponent * math.log(rate) + math.log(2))

                def integrand(radius):
                    if radius == 0:
                        if 2 * exponent != 1:
                            return 0.
                        log_numerator = log_prefactor
                    else:
                        log_numerator = log_prefactor + (2 * exponent - 1) * math.log(radius) - radius * radius
                    if log_numerator < -740:
                        return 0.
                    value = log_numerator - float(_relative_spectrum_log(radius * radius / rate, parameters))
                    return math.exp(value)

                value, error = quad(integrand, 0., np.inf, epsabs=2e-12, epsrel=2e-12, limit=200)
                moments[power, j, index] = moments[power, index, j] = value
                errors[power, j, index] = errors[power, index, j] = error
    return _moments_to_blocks(moments, parameters), errors


def _symmetric_bases(dimension):
    basis = []
    for column in range(dimension):
        for row in range(column):
            matrix = np.zeros((dimension, dimension))
            matrix[row, column] = matrix[column, row] = 1 / math.sqrt(2)
            basis.append(matrix)
    for count in range(1, dimension):
        matrix = np.zeros((dimension, dimension))
        matrix[np.arange(count), np.arange(count)] = 1 / math.sqrt(count * (count + 1))
        matrix[count, count] = -count / math.sqrt(count * (count + 1))
        basis.append(matrix)
    return np.asarray(basis).reshape(-1, dimension, dimension)


def _svec(matrix):
    return np.array([matrix[row, column] * (1 if row == column else math.sqrt(2))
                     for column in range(len(matrix)) for row in range(column + 1)])


@dataclass(frozen=True)
class SingleQueryMultiscaleField:
    dimension: int
    bandwidths: np.ndarray
    h_min: float
    h_max: float
    target_blocks: dict
    reference_blocks: dict
    covariance_blocks: dict
    factors: dict
    noise_coefficients: dict
    metadata: dict
    traceless_basis: np.ndarray



def prepare_single_query_field(dimension, bandwidths, *, h_min, h_max,
                               numerical_margin=1e-10, radial_bins=32768,
                               quadrature_order=48, mixture_order=96,
                               audit_tolerance=2e-8):
    """Prepare the original common spectrum's conservative shell field.

    All inputs and all metadata are public. The right-endpoint lower relative
    spectrum proves K_original <= K_shell <= Gamma*K_original in exact reals.
    The floating evaluations of that construction remain explicitly audited,
    not interval-certified. Cholesky is attempted once with a declared margin.
    """
    hs, parameters = _spectrum_parameters(dimension, bandwidths, h_min, h_max)
    dimension = parameters["D"]
    margin = _scalar(numerical_margin, "numerical_margin", positive=True)
    bins = _positive_integer(radial_bins, "radial_bins")
    order = _positive_integer(quadrature_order, "quadrature_order")
    mixture_order = _positive_integer(mixture_order, "mixture_order")
    tolerance = _scalar(audit_tolerance, "audit_tolerance", positive=True)
    if bins < 2:
        raise ValueError("radial_bins must be at least two")
    count, tail_u = len(hs), float(80 + dimension)
    metadata = dict(mechanism="single_query_common_multiscale_gaussian_field",
                    coordinate_system="q,v,svec(B)/sqrt(2)", query_count=1,
                    dimension=dimension, bandwidths=hs.tolist(), h_min=float(h_min), h_max=float(h_max),
                    log_bandwidth_range=parameters["L"], c_D=parameters["c"],
                    endpoint_weight=parameters["b"], Z=parameters["Z"],
                    ideal_covariance_dominance_proved=True,
                    covariance_dominance_certified=False, floating_point_dp_certified=False,
                    sensitivity_numerator=math.sqrt(2), cholesky_attempts=1,
                    quadrature_order=order, mixture_order=mixture_order,
                    audit_tolerance=tolerance, numerical_margin_requested=margin,
                    cb_kind="public_nominal_scale_not_a_probability_bound")
    if parameters["L"] == 0:
        target = dict(scalar_trace=np.eye(2), vector=np.eye(1), traceless=np.eye(1))
        reference = {key: value.copy() for key, value in target.items()}
        discrepancies = {key: 0. for key in _BLOCKS}
        tails = {key: 0. for key in _BLOCKS}
        metadata.update(integration_method="analytic_single_scale", independent_method="analytic_single_scale",
                        audited_pairs=1, audited_radial_moments=0, spectral_inflation_bound=1.,
                        radial_shell_bins=0, radial_tail_u=tail_u, maximum_adaptive_error_estimate=0.)
    else:
        moments, shell = _shell_moments(parameters, bins, tail_u)
        target = _moments_to_blocks(moments, parameters)
        reference, tails, node_count = _original_gram(parameters, order, mixture_order, tail_u)
        adaptive, errors = _original_adaptive(parameters)
        discrepancies = {key: float(np.linalg.norm(reference[key] - adaptive[key], ord=np.inf)) for key in _BLOCKS}
        for key in _BLOCKS:
            if discrepancies[key] > tolerance * max(1., float(np.linalg.norm(reference[key], ord=np.inf))):
                raise ArithmeticError(f"independent original-spectrum integral audit failed for {key}")
        # Preserve complete public endpoints as lists for reproducible audits.
        metadata.update({key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in shell.items()})
        metadata.update(integration_method="monotone_relative_spectrum_shell",
                        independent_method="adaptive_radial_incomplete_gamma",
                        original_gram_method="positive_composite_legendre_gram",
                        original_gram_nodes=node_count, audited_pairs=count * (count + 1) // 2,
                        audited_radial_moments=3 * count * (count + 1) // 2,
                        maximum_adaptive_error_estimate=float(errors.max()),
                        adaptive_error_estimates=errors.tolist())
    metadata["block_discrepancies"] = discrepancies
    metadata["maximum_block_discrepancy"] = max(discrepancies.values())
    metadata["original_radial_tail_operator_bounds"] = tails
    norm = max(float(np.linalg.norm(value, ord=np.inf)) for value in target.values())
    margin = max(margin, 64 * np.finfo(float).eps * (2 * count) * max(1., norm))
    metadata["numerical_diagonal_margin"] = margin
    factors, actual, eigenvalues, residuals, increments = {}, {}, {}, {}, {}
    relative_costs = {}
    for key in _BLOCKS:
        target[key] = (target[key] + target[key].T) / 2
        matrix = target[key] + margin * np.eye(len(target[key]))
        eigenvalues[key] = float(np.linalg.eigvalsh(matrix).min())
        if eigenvalues[key] <= 0 or not np.isfinite(matrix).all():
            raise ArithmeticError(f"declared covariance margin did not produce a positive {key} block")
        try:
            factors[key] = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ArithmeticError(f"single-attempt Cholesky failed for {key}") from error
        actual[key] = factors[key] @ factors[key].T
        residuals[key] = float(np.linalg.norm(actual[key] - matrix, ord=np.inf))
        increments[key] = float(np.linalg.eigvalsh(target[key] - reference[key]).min())
        relative_costs[key] = float(np.max(np.sqrt(np.diag(target[key]) / np.diag(reference[key]))))
    metadata.update(minimum_eigenvalues=eigenvalues, factor_residuals=residuals,
                    shell_minus_original_minimum_eigenvalues=increments,
                    maximum_marginal_sd_ratios=relative_costs)
    scalar, vector, traceless = (actual[key] for key in _BLOCKS)
    a0, a2, a4 = np.diag(scalar)[:count], np.diag(vector), np.diag(traceless)
    trace_variance = np.diag(scalar)[count:]
    gamma = 2 * (trace_variance - a4) / dimension
    cb = (np.sqrt(2 * trace_variance) if dimension == 1
          else 2 * np.sqrt(dimension * a4) + np.sqrt(np.maximum(gamma, 0.)))
    coefficients = dict(c0=np.sqrt(a0), cv=np.sqrt(dimension * a2), cb=cb,
                        a0=a0, a2=a2, a4=a4, gamma=gamma, trace_variance=trace_variance,
                        B_direction_variance=2 * a4 + gamma)
    fingerprint = hashlib.sha256()
    fingerprint.update(np.asarray([dimension, h_min, h_max, margin], dtype=float).tobytes())
    fingerprint.update(hs.tobytes())
    for key in _BLOCKS:
        fingerprint.update(factors[key].tobytes())
    metadata["public_design_hash"] = fingerprint.hexdigest()
    return SingleQueryMultiscaleField(
        dimension, _readonly(hs), float(h_min), float(h_max),
        {key: _readonly(value) for key, value in target.items()},
        {key: _readonly(value) for key, value in reference.items()},
        {key: _readonly(value) for key, value in actual.items()},
        {key: _readonly(value) for key, value in factors.items()},
        {key: _readonly(value) for key, value in coefficients.items()}, metadata,
        _readonly(_symmetric_bases(dimension)),
    )
