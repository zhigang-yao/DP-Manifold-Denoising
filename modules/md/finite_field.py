"""Explicit finite Gaussian spectrum using the canonical derivative kernel.

The archived feasible H9 design is g=H9,w=1/9, not a claim of an optimized
spectrum. Exact-real privacy follows from the positive normalized mixture;
quadrature/jitter remain numerical ideal-mechanism simulation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, roots_legendre

from modules.md._kernel import MixtureKernel


class FiniteSpectrumKernel(MixtureKernel):
    def __init__(self, dimension, bandwidths, order=128, *, scales=None, weights=None):
        self.d = int(dimension)
        self.hs = np.asarray(bandwidths, dtype=float)
        self.gs = np.asarray(self.hs if scales is None else scales, dtype=float)
        self.weights = np.asarray(np.ones(len(self.gs))/len(self.gs) if weights is None else weights, dtype=float)
        if (self.d not in (2, 3) or self.hs.ndim != 1 or not len(self.hs)
                or not np.isfinite(self.hs).all() or np.any(self.hs <= 0)
                or np.any(np.diff(self.hs) <= 0)
                or self.gs.ndim != 1 or self.weights.shape != self.gs.shape
                or not np.isfinite(self.gs).all() or np.any(self.gs <= 0)
                or not np.isfinite(self.weights).all() or np.any(self.weights < 0)
                or not np.isclose(self.weights.sum(), 1., rtol=0, atol=1e-14)):
            raise ValueError('invalid public finite-spectrum design')
        active = self.weights > 0
        self.gs, self.weights = self.gs[active], self.weights[active]
        self.hmin = float(self.gs.min())
        if self.hmin >= np.sqrt(2)*self.hs.min():
            raise ValueError('finite spectrum fails strict tail admissibility')
        if isinstance(order, bool) or int(order) != order or order < 8:
            raise ValueError('quadrature order must be an integer >=8')
        self.order = int(order)
        self.edges = (0., .0625, .125, .25, .5, 1., 2., 4., 8., 12., 16.)
        z, w = roots_legendre(self.order)
        self._rho = np.concatenate([.5*(b-a)*z+.5*(a+b) for a,b in zip(self.edges[:-1], self.edges[1:])])
        self._weights = np.concatenate([.5*(b-a)*w for a,b in zip(self.edges[:-1], self.edges[1:])])
        self._frequency = self._rho/self.hmin
        ratios = self.gs/self.hmin
        self._log_relative = logsumexp(np.log(self.weights)[None, :] + self.d*np.log(ratios)[None, :]
            - .5*(ratios**2-1)[None, :]*self._rho[:, None]**2, axis=1)
        self.p = {'Z': 1.}  # Parent radial weights integrate rho_h*rho_b/rho_w.
        hs = sorted(map(float, self.hs))
        self._pairs = [(h,k) for i,h in enumerate(hs) for k in hs[i:]]
        self._pair_index = {pair:i for i,pair in enumerate(self._pairs)}
        self._shared_distances, self._shared_values, self._shared_weights = None, [], {}
        self.design = dict(mode='finite_joint', dimension=self.d, hs=self.hs.tolist(),
            scales=self.gs.tolist(), weights=self.weights.tolist(), order=self.order,
            radial_edges=list(self.edges), packing='natural_upper_triangle_column_v1')
        fingerprint = hashlib.sha256(json.dumps(self.design, sort_keys=True).encode())
        for path in [Path(__file__), Path(__file__).with_name('_kernel.py'), Path(__file__).with_name('gaussian_field.py')]:
            fingerprint.update(hashlib.sha256(path.read_bytes()).digest())
        self.design_digest = fingerprint.digest()
