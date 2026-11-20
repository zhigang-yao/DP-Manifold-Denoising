"""Frozen mixture quadrature and derivative operators."""

from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre, jv, gammaln, j0, j1, spherical_jn
from itertools import islice
from modules.md.multiscale_gaussian_field import (
    _spectrum_parameters,
    _relative_spectrum_log,
)


def _operators(d, h, kind):
    unit = lambda i: tuple(1 if j == i else 0 for j in range(d))
    zero = (0,) * d
    if kind == "density":
        return [[(zero, 1.0)]]
    out = [[(zero, 1.0)]] + [
        [(unit(i), h if kind == "moments" else 1.0)] for i in range(d)
    ]
    if kind == "moments":
        for j in range(d):
            for i in range(j + 1):
                a = list(zero)
                a[i] += 1
                a[j] += 1
                terms = [(tuple(a), h * h / (np.sqrt(2) if i == j else 1.0))]
                if i == j:
                    terms.append((zero, 1 / np.sqrt(2)))
                out.append(terms)
    return out


@lru_cache(None)
def _pairings(indices):
    if not indices:
        return [((), ())]
    first, *rest = indices
    out = []
    for pairs, singles in _pairings(tuple(rest)):
        out.append((pairs, (first,) + singles))
    for pos, other in enumerate(rest):
        if first == other:
            remaining = tuple(rest[:pos] + rest[pos + 1 :])
            for pairs, singles in _pairings(remaining):
                out.append((((first, other),) + pairs, singles))
    return out


def _partial(radials, r, orders):
    indices = tuple(i for i, n in enumerate(orders) for _ in range(n))
    out = np.zeros(r.shape[:-1])
    for pairs, singles in _pairings(indices):
        term = radials[len(indices) - len(pairs)].copy()
        for i in singles:
            term *= r[..., i]
        out += term
    return out


class MixtureKernel:
    """Frozen quadrature, native angular identities and one exact-radius cache."""

    def __init__(self, dimension, bandwidths, order=256, spectrum_range=(0.025, 0.4)):
        self.d = dimension
        self.hs = np.array(bandwidths)
        self.hmin = spectrum_range[0]
        _, self.p = _spectrum_parameters(dimension, self.hs, *spectrum_range)
        self.order = order
        z, w = roots_legendre(self.order)
        edges = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)
        self._rho = np.concatenate(
            [0.5 * (b - a) * z + 0.5 * (b + a) for a, b in zip(edges[:-1], edges[1:])]
        )
        self._weights = np.concatenate(
            [0.5 * (b - a) * w for a, b in zip(edges[:-1], edges[1:])]
        )
        self._frequency = self._rho / self.hmin
        self._log_relative = _relative_spectrum_log(self._rho * self._rho / 2, self.p)
        hs = sorted(set(map(float, self.hs)))
        self._pairs = [(h, k) for i, h in enumerate(hs) for k in hs[i:]]
        self._pair_index = {pair: i for i, pair in enumerate(self._pairs)}
        self._shared_distances = None
        self._shared_values = []
        self._shared_weights = {}

    def _generic_radials(self, r, h, k, maximum=4):
        # Segmenting resolves both central mass and oscillatory distant queries.
        z, w = roots_legendre(self.order)
        edges = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
        rho = np.concatenate(
            [0.5 * (b - a) * z + 0.5 * (b + a) for a, b in zip(edges[:-1], edges[1:])]
        )
        weights = np.concatenate(
            [0.5 * (b - a) * w for a, b in zip(edges[:-1], edges[1:])]
        )
        beta = (h / self.hmin) ** 2
        beta2 = (k / self.hmin) ** 2
        D = self.d
        shape = D / 2
        logw = (
            np.log(weights)
            + np.log(self.p["Z"])
            + shape * np.log(beta * beta2)
            + (D - 1) * np.log(rho)
            - (beta + beta2 - 1) * rho * rho / 2
            - (shape - 1) * np.log(2)
            - gammaln(shape)
            - _relative_spectrum_log(rho * rho / 2, self.p)
        )
        wr = np.exp(logw)
        distance = np.linalg.norm(r, axis=-1).ravel()
        out = [np.empty_like(distance) for _ in range(maximum + 1)]
        frequency = rho / self.hmin
        # Bounded scratch memory, exact same public quadrature for all reads.
        for start in range(0, len(distance), 256):
            rr = distance[start : start + 256]
            s = rr[:, None] * frequency
            denom = 1.0
            for m in range(maximum + 1):
                nu = shape + m - 1
                angular = np.ones_like(s)
                mask = s > 1e-7
                angular[mask] = np.exp(
                    gammaln(shape + m) + nu * np.log(2 / s[mask])
                ) * jv(nu, s[mask])
                if m:
                    denom *= D + 2 * (m - 1)
                out[m][start : start + len(rr)] = angular @ (
                    wr * ((-1.0) ** m) * frequency ** (2 * m) / denom
                )
        return [v.reshape(r.shape[:-1]) for v in out]

    def covariance(self, x, y, h, k, kind_left="moments", kind_right="moments"):
        r = x[:, None, :] - y[None, :, :]
        left = _operators(self.d, h, kind_left)
        right = _operators(self.d, k, kind_right)
        maximum = max(sum(o) for terms in left for o, c in terms) + max(
            sum(o) for terms in right for o, c in terms
        )
        rad = self.radials(r, h, k, maximum)
        out = np.empty((len(x), len(left), len(y), len(right)))
        partials = {}
        for i, li in enumerate(left):
            for j, rj in enumerate(right):
                block = np.zeros((len(x), len(y)))
                for a, ca in li:
                    for b, cb in rj:
                        order = tuple(aa + bb for aa, bb in zip(a, b))
                        if order not in partials:
                            partials[order] = _partial(rad, r, order)
                        block += ca * cb * ((-1.0) ** sum(b)) * partials[order]
                out[:, i, :, j] = block
        return out.reshape(len(x) * len(left), len(y) * len(right))

    @lru_cache(maxsize=256)
    def _radial_weights(self, h, k, maximum):
        beta = (h / self.hmin) ** 2
        beta2 = (k / self.hmin) ** 2
        shape = self.d / 2
        logw = (
            np.log(self._weights)
            + np.log(self.p["Z"])
            + shape * np.log(beta * beta2)
            + (self.d - 1) * np.log(self._rho)
            - (beta + beta2 - 1) * self._rho * self._rho / 2
            - (shape - 1) * np.log(2)
            - gammaln(shape)
            - self._log_relative
        )
        wr = np.exp(logw)
        denom = 1.0
        answer = []
        for m in range(maximum + 1):
            if m:
                denom *= self.d + 2 * (m - 1)
            answer.append(wr * ((-1.0) ** m) * self._frequency ** (2 * m) / denom)
        return answer

    @staticmethod
    def _angular_2d(s, maximum):
        """Return 2**m*m!*J_m(s)/s**m, with its continuous value at zero.

        Upward recurrence is used only for s>12, safely beyond all orders here.
        Small arguments retain scipy's general J_m evaluation, avoiding loss
        from subtractive cancellation. J0/J1 use SciPy's specialized routines.
        """
        b0 = j0(s)
        b0[s <= 1e-7] = 1.0
        yield b0
        if maximum == 0:
            return
        b1 = j1(s)
        mask = s > 1e-7
        angular = np.ones_like(s)
        np.divide(2 * b1, s, out=angular, where=mask)
        yield angular
        large = s > 12.0
        for m in range(2, maximum + 1):
            b2 = np.empty_like(s)
            b2[large] = 2 * (m - 1) / s[large] * b1[large] - b0[large]
            b2[~large] = jv(m, s[~large])
            angular = np.ones_like(s)
            # Match the original quadrature's small-argument cutoff exactly.
            angular[mask] = np.exp(gammaln(m + 1) + m * np.log(2 / s[mask])) * b2[mask]
            yield angular
            b0, b1 = b1, b2

    @staticmethod
    def _angular_3d(s, maximum):
        """Yield (2m+1)!! j_m(s)/s**m with the original <=1e-7 cutoff.

        scipy.special.spherical_jn uses stable small-argument evaluations and
        ascending recurrence for real arguments larger than the order. The
        normalization equals the original half-integer Bessel-J expression.
        """
        mask = s > 1e-7
        coefficient = 1.0
        for m in range(maximum + 1):
            if m:
                coefficient *= 2 * m + 1
            angular = np.ones_like(s)
            angular[mask] = coefficient * spherical_jn(m, s[mask]) / s[mask] ** m
            yield angular

    def _angular_native(self, s, maximum):
        if self.d == 3:
            return self._angular_3d(s, maximum)
        if self.d == 4:
            # The D4 angular function of radial order m is exactly the D2
            # function of order m+1. The inherited large-s recurrence is safe
            # for all required integer orders 1..5 (s>12); smaller s uses jv.
            return islice(self._angular_2d(s, maximum + 1), 1, None)
        return self._angular_2d(s, maximum)

    def radials(self, r, h, k, maximum=4):
        key = (float(min(h, k)), float(max(h, k)))
        if self.d not in (2, 3, 4) or maximum > 4 or key not in self._pair_index:
            return self._generic_radials(r, h, k, maximum)
        r = np.asarray(r)
        shape = r.shape[:-1]
        distance, inverse = np.unique(
            np.linalg.norm(r, axis=-1).ravel(), return_inverse=True
        )
        same = (
            self._shared_distances is not None
            and np.array_equal(distance, self._shared_distances)
            and len(self._shared_values) > maximum
        )
        if not same:
            if maximum not in self._shared_weights:
                all_weights = [
                    self._radial_weights(hh, kk, maximum) for hh, kk in self._pairs
                ]
                self._shared_weights[maximum] = [
                    np.column_stack([weights[m] for weights in all_weights])
                    for m in range(maximum + 1)
                ]
            weights = self._shared_weights[maximum]
            answer = [
                np.empty((len(distance), len(self._pairs))) for _ in range(maximum + 1)
            ]
            for start in range(0, len(distance), 256):
                stop = min(start + 256, len(distance))
                s = distance[start:stop, None] * self._frequency
                for m, angular in enumerate(self._angular_native(s, maximum)):
                    answer[m][start:stop] = angular @ weights[m]
            self._shared_distances = distance
            self._shared_values = answer
        index = self._pair_index[key]
        return [
            values[inverse, index].reshape(shape)
            for values in self._shared_values[: maximum + 1]
        ]
