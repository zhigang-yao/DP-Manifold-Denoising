"""experiments._shared.case4_release with three runtime speed-ups (2026-11-20); the frozen modules are not edited.
1. HighDFiniteKernel.angular evaluates hyp0f1 on DPMD_ANGULAR_THREADS threads (elementwise, bit-identical).
2. SharedStructuredField.coeffs (correlated-bandwidth branch) evaluates its 256-row chunks on the same
   threads (each chunk is computed exactly as before, bit-identical).
3. The terminal Schur check in SharedStructuredField.conditional_masses used a full eigvalsh of a UxU matrix
   only to test eigmin<=0. With DPMD_CHOLESKY_FIRST=1 a Cholesky factor is tried first: if it succeeds the
   matrix is taken as positive definite (repair=0) and the same factor is reused for the conditional draw;
   if it fails the original eigvalsh/repair path runs. Identical whenever eigvalsh would report eigmin>0.
usage: same arguments as experiments._shared.case4_release"""
import os
import types
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.linalg import cholesky
from scipy.special import hyp0f1
from modules.md.highd_kernel import HighDFiniteKernel
import modules.md.structured_shared_field as ssf
from experiments._shared import case4_release

THREADS = int(os.environ.get('DPMD_ANGULAR_THREADS', '1'))
_pool = ThreadPoolExecutor(THREADS) if THREADS > 1 else None


def angular(s, D, maximum):
    z = -0.25 * s * s
    if _pool is None or z.shape[0] < 2 * THREADS:
        return [hyp0f1(D / 2 + m, z) for m in range(maximum + 1)]
    rows = np.array_split(np.arange(z.shape[0]), THREADS)
    parts = list(_pool.map(lambda job: hyp0f1(D / 2 + job[0], z[job[1]]),
                           [(m, r) for m in range(maximum + 1) for r in rows]))
    return [np.concatenate(parts[m * THREADS:(m + 1) * THREADS]) for m in range(maximum + 1)]


HighDFiniteKernel.angular = staticmethod(angular)

_coeffs = ssf.SharedStructuredField.coeffs


def coeffs(self, distances, maximum=4, chunk=256):
    if _pool is None or self.bandwidth_correlation == "independent":
        return _coeffs(self, distances, maximum, chunk)
    if maximum < 0 or maximum > 4: raise ValueError("radial derivative order must be 0..4")
    distance = np.asarray(distances, float)
    flat = distance.ravel()
    ans = np.empty((len(flat), self.H * self.H, maximum + 1))

    def work(start):
        end = min(len(flat), start + chunk)
        freq = flat[start:end, None] * self._freq[None, :]
        z = -.25 * freq * freq
        for k in range(maximum + 1):
            phi = hyp0f1(self.D / 2 + k, z)
            ans[start:end, :, k] = phi @ self._radial_weights[k]
    list(_pool.map(work, range(0, len(flat), chunk)))
    return ans.reshape(distance.shape + (self.H, self.H, maximum + 1))


ssf.SharedStructuredField.coeffs = coeffs

if os.environ.get('DPMD_CHOLESKY_FIRST') == '1':
    _last = {}

    def _eigvalsh(a, UPLO='L'):
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[0] >= 512:
            try:
                root = cholesky(a, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                _last.clear()
                return np.linalg.eigvalsh(a, UPLO)
            _last['a'], _last['root'] = a, root
            return np.array([np.inf])
        return np.linalg.eigvalsh(a, UPLO)

    def _cholesky(a, *args, **kwargs):
        if 'a' in _last and _last['a'].shape == np.shape(a) and np.array_equal(_last['a'], a):
            root = _last.pop('root'); _last.clear()
            return root
        return cholesky(a, *args, **kwargs)

    class _Numpy(types.ModuleType):
        def __getattr__(self, name):
            return getattr(np, name)

    proxy = _Numpy('numpy_cholesky_first')
    proxy.linalg = types.SimpleNamespace(**{k: getattr(np.linalg, k) for k in dir(np.linalg) if not k.startswith('_')})
    proxy.linalg.eigvalsh = _eigvalsh
    ssf.np = proxy
    ssf.cholesky = _cholesky

class ChebyshevTable:
    """Piecewise Chebyshev interpolant of exact(d) -> [len(d), channels] on panels of the given width (2026-11-20).
    Nodes are evaluated with the exact (frozen) radial code; the table grows when a larger distance appears."""

    def __init__(self, exact, width, nodes=24):
        self.exact, self.width, self.n = exact, float(width), nodes
        j = np.arange(nodes)
        self.t = np.cos(np.pi * (j + .5) / nodes)
        self.dct = np.cos(np.pi * np.outer(j, j + .5) / nodes) * (2 / nodes); self.dct[0] /= 2
        self.coef, self.panels = None, 0

    def _build(self, dmax):
        self.panels = int(np.ceil(dmax / self.width)) + 1
        x = (np.arange(self.panels)[:, None] * self.width + self.width * (self.t[None, :] + 1) / 2).ravel()
        values = self.exact(x).reshape(self.panels, self.n, -1)
        self.coef = np.einsum('kj,pjc->pkc', self.dct, values)

    def __call__(self, d):
        d = np.asarray(d, float).ravel()
        if self.coef is None or d.max(initial=0.) >= self.panels * self.width:
            self._build(max(1.25 * d.max(initial=0.), self.width))
        p = np.minimum((d // self.width).astype(np.int64), self.panels - 1)
        t = 2 * (d - p * self.width) / self.width - 1
        out = np.empty((len(d), self.coef.shape[2]))
        order = np.argsort(p, kind='stable'); ps = p[order]
        cut = np.flatnonzero(np.diff(ps)) + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(d)]):
            idx = order[s:e]; tt = t[idx]
            T = np.empty((len(idx), self.n)); T[:, 0] = 1; T[:, 1] = tt
            for k in range(2, self.n): T[:, k] = 2 * tt * T[:, k - 1] - T[:, k - 2]
            out[idx] = T @ self.coef[ps[s]]
        return out


if os.environ.get('DPMD_RADIAL_INTERP') == '1':
    # Not bit-identical: radial covariances from piecewise Chebyshev tables of the exact radial functions
    # (panel width h_min/4, 24 nodes); accuracy is checked against the exact code before use.
    _generic = HighDFiniteKernel.generic_radials
    _tables = {}

    def generic_radials(self, r, h, k, maximum=4):
        key = (float(min(h, k)), float(max(h, k)))
        if key not in self._pair_index:
            raise ValueError('bandwidth pair outside public design')
        r = np.asarray(r); shape = r.shape[:-1]
        tid = (id(self), key, maximum)
        if tid not in _tables:
            _tables[tid] = ChebyshevTable(lambda x: np.stack(_generic(self, x[:, None], h, k, maximum), axis=1),
                                          min(key) / 4)
        values = _tables[tid](np.linalg.norm(r, axis=-1))
        return [values[:, m].reshape(shape) for m in range(maximum + 1)]

    HighDFiniteKernel.generic_radials = generic_radials
    _threaded_coeffs = ssf.SharedStructuredField.coeffs

    def interp_coeffs(self, distances, maximum=4, chunk=256):
        if self.bandwidth_correlation == "independent":
            return _threaded_coeffs(self, distances, maximum, chunk)
        distance = np.asarray(distances, float)
        tid = (id(self), maximum)
        if tid not in _tables:
            _tables[tid] = ChebyshevTable(
                lambda x: _threaded_coeffs(self, x, maximum, chunk).reshape(len(x), -1), float(np.min(self.hs)) / 4)
        return _tables[tid](distance).reshape(distance.shape + (self.H, self.H, maximum + 1))

    ssf.SharedStructuredField.coeffs = interp_coeffs

if __name__ == '__main__':
    case4_release.main()
