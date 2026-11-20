"""D-general exact finite radial covariance, adapted from EXP-G source.
This module only defines a covariance kernel; release semantics live in backends.py.
"""
import hashlib, json, math, numpy as np
from pathlib import Path
from scipy.special import logsumexp, hyp0f1, roots_legendre
from modules.md.finite_field import FiniteSpectrumKernel
from modules.md._kernel import _operators
VERSION = "G_per_scale_v1"

class HighDFiniteKernel(FiniteSpectrumKernel):
    """FiniteSpectrumKernel without the D in (2,3) guard; identical construction otherwise."""

    def __init__(self, dimension, bandwidths, order=128, *, scales=None, weights=None):
        # Construct as the frozen class at D=3, then rebuild every D-dependent quantity at the true D
        # with the frozen formulas (tested to reproduce the frozen object exactly at D=2,3).
        super().__init__(3, bandwidths, order, scales=scales, weights=weights)
        self.d = int(dimension)
        if self.d < 2:
            raise ValueError('ambient dimension must be >=2')
        # In large D the Gaussian radial chi_D shell concentrates near
        # sqrt(D). A fixed cutoff rho=16 loses almost all spectral mass
        # by D=400, despite well-conditioned finite covariance formulas.
        # Preserve the original 0..16 rule exactly for D<=4; otherwise
        # append enough public quadrature segments to cover sqrt(D)+14.
        if self.d>=5:
            upper=max(16.0,float(np.sqrt(self.d)+14.0))
            edges=list(self.edges)
            edges.extend(float(v) for v in np.arange(24.,upper,8.))
            if upper>edges[-1]:
                edges.append(upper)
            self.edges=tuple(edges)
            nodes,weights=roots_legendre(self.order)
            self._rho=np.concatenate([
                .5*(right-left)*nodes+.5*(left+right)
                for left,right in zip(edges[:-1],edges[1:])])
            self._weights=np.concatenate([
                .5*(right-left)*weights
                for left,right in zip(edges[:-1],edges[1:])])
            self._frequency=self._rho/self.hmin
            self._radial_weights.cache_clear()
            self._shared_distances,self._shared_values=None,[]
            self._shared_weights={}
        ratios = self.gs / self.hmin
        self._log_relative = logsumexp(np.log(self.weights)[None, :] + self.d*np.log(ratios)[None, :]
            - .5*(ratios**2-1)[None, :]*self._rho[:, None]**2, axis=1)
        self._radial_weights.cache_clear()
        self.design = dict(self.design, dimension=self.d, backend=VERSION,
                           radial_edges=list(self.edges),
                           radial_upper=float(self.edges[-1]),
                           shell_coverage_policy="max(16,sqrt(D)+14) for D>=5")
        fingerprint = hashlib.sha256(json.dumps(self.design, sort_keys=True).encode())
        fingerprint.update(self.design_digest)
        fingerprint.update(hashlib.sha256(Path(__file__).read_bytes()).digest())
        self.design_digest = fingerprint.digest()

    @staticmethod
    def angular(s, D, maximum):
        """A_{D,m}(s) for m=0..maximum; equals the native D2/D3/D4 identities (tested)."""
        z = -0.25 * s * s
        return [hyp0f1(D / 2 + m, z) for m in range(maximum + 1)]

    def radials(self, r, h, k, maximum=4):
        if self.d in (2, 3, 4):
            return super().radials(r, h, k, maximum)
        return self.generic_radials(r, h, k, maximum)

    def generic_radials(self, r, h, k, maximum=4):
        key = (float(min(h, k)), float(max(h, k)))
        if key not in self._pair_index:
            raise ValueError('bandwidth pair outside public design')
        r = np.asarray(r)
        shape = r.shape[:-1]
        distance, inverse = np.unique(np.linalg.norm(r, axis=-1).ravel(), return_inverse=True)
        weights = self._radial_weights(*key, maximum)
        out = [np.empty(len(distance)) for _ in range(maximum + 1)]
        for start in range(0, len(distance), 256):
            stop = min(start + 256, len(distance))
            s = distance[start:stop, None] * self._frequency
            for m, a in enumerate(self.angular(s, self.d, maximum)):
                out[m][start:stop] = a @ weights[m]
        return [v[inverse].reshape(shape) for v in out]


class PerScaleKernel:
    """Independent single-atom finite field per bandwidth; zero cross-bandwidth covariance."""

    def __init__(self, dimension, hs, order=128):
        self.d, self.hs, self.order = int(dimension), np.asarray(hs, float), int(order)
        self.kernels = {float(h): HighDFiniteKernel(dimension, [float(h)], order, scales=[float(h)], weights=[1.])
                        for h in self.hs}
        self.design = dict(mode='per_scale_finite', backend=VERSION, dimension=self.d, hs=self.hs.tolist(),
                           per_bandwidth=dict(scales='g=h', weights=[1.]), order=self.order,
                           cross_bandwidth_covariance=0, sd_multiplier=math.sqrt(len(self.hs)))
        digest = hashlib.sha256(json.dumps(self.design, sort_keys=True).encode())
        for h in self.hs:
            digest.update(self.kernels[float(h)].design_digest)
        self.design_digest = digest.digest()

    def covariance(self, x, y, h, k, kind_left='moments', kind_right='moments'):
        if float(h) != float(k):
            from modules.md._kernel import _operators
            return np.zeros((len(x) * len(_operators(self.d, h, kind_left)), len(y) * len(_operators(self.d, k, kind_right))))
        return self.kernels[float(h)].covariance(x, y, h, k, kind_left, kind_right)


