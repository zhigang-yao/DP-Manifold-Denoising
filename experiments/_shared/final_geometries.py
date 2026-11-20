"""Data for the final campaign (2026-11-08) beyond modules.synthetic.

- Further geometries of the paper appendix: the flat 2-plane and the catenoid (2026-11-18; the circle is
  modules.synthetic's). The three-dimensional Swiss roll (generate_swiss3) was run and replaced by the catenoid. The affine segment, figure-eight and helix of the 2026-11-08 design are
  kept for the campaign manifest but no longer reported.
- The concentrated bounded noise law: uniform direction, radius density proportional to 1-r/sigma on [0,sigma].
- The native D=10 Fourier torus of experiments/fourier_torus.py.

Synthetic geometries keep modules.synthetic.generate unchanged; only the reference noise is redrawn for the
central law (same reference-noise stream, so clean points and queries are those of the uniform-ball design).
"""
import numpy as np

from geometry.synthetic import SyntheticConfig, generate

# Seed ids are 101 + SEED_ORDER.index. Ellipse, flat 3-ball and Fourier 3-torus were piloted on 2026-11-17/18
# and are not reported; they stay in the order so that flat2, swiss3 and catenoid keep the seeds of their runs.
SEED_ORDER = ('affine', 'figure_eight', 'helix', 'ellipse', 'flat3', 'flat2', 'swiss3', 'fourier3', 'catenoid')
EXTRA = SEED_ORDER[:3]  # the further geometries of campaign_final's manifest (2026-11-08)
SEED = 20261105
CATENOID_C, CATENOID_H = .3, .5  # waist radius (= reach) and half height


def _rng(geometry, replicate, role):
    return np.random.default_rng(np.random.SeedSequence([SEED, 101 + SEED_ORDER.index(geometry), replicate, role]))


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _curve(geometry, t):
    """Native points and unit tangents of a curve at parameters t."""
    if geometry == 'affine':
        p = np.stack([t, 0 * t, 0 * t], axis=1); v = np.stack([1 + 0 * t, 0 * t, 0 * t], axis=1)
    elif geometry == 'figure_eight':
        p = np.stack([np.sin(t), np.sin(t) * np.cos(t)], axis=1); v = np.stack([np.cos(t), np.cos(2 * t)], axis=1)
    else:
        p = np.stack([.5 * np.cos(t), .5 * np.sin(t), .4 * t / np.pi], axis=1)
        v = np.stack([-.5 * np.sin(t), .5 * np.cos(t), .4 / np.pi + 0 * t], axis=1)
    return p, _unit(v)


def _speed(geometry, t):
    if geometry == 'figure_eight':
        return np.sqrt(np.cos(t) ** 2 + np.cos(2 * t) ** 2)
    return np.ones_like(t)


def _sample_t(geometry, count, rng, *, query):
    # arc-length-uniform by rejection on the speed; query base points keep away from endpoints / the crossing
    low, high = dict(affine=(-1., 1.), figure_eight=(0., 2 * np.pi), helix=(-2 * np.pi, 2 * np.pi))[geometry]
    helix_margin = .1 / np.hypot(.5, .4 / np.pi)  # arc length 0.1 in parameter units
    out = []
    while sum(map(len, out)) < count:
        t = rng.uniform(low, high, max(512, count))
        keep = rng.uniform(size=len(t)) < _speed(geometry, t) / np.sqrt(2)
        t = t[keep]
        if query:
            if geometry == 'affine': t = t[np.abs(t) <= .8]
            if geometry == 'helix': t = t[np.abs(t) <= 2 * np.pi - helix_margin]
            if geometry == 'figure_eight': t = t[np.linalg.norm(_curve(geometry, t)[0], axis=1) >= .35]
        out.append(t)
    return np.concatenate(out)[:count]


def generate_extra(geometry, n, Q, D, sigma, a, replicate):
    t = _sample_t(geometry, n, _rng(geometry, replicate, 1), query=False)
    clean, _ = _curve(geometry, t)
    native = clean.shape[1]
    emb = np.linalg.qr(_rng(geometry, 0, 7).standard_normal((D, native)))[0]
    latent, tangent = _curve(geometry, _sample_t(geometry, Q, _rng(geometry, 0, 3), query=True))
    latent, tangent = latent @ emb.T, tangent @ emb.T
    g = _rng(geometry, 0, 4).standard_normal((Q, D))
    normals = _unit(g - np.sum(g * tangent, axis=1, keepdims=True) * tangent)
    from geometry.noise import sample_uniform_ball_noise
    reference = clean @ emb.T
    noise = sample_uniform_ball_noise(n, D, sigma, _rng(geometry, replicate, 2))
    return dict(reference=reference + noise, clean_reference=reference, queries=latent + a * normals,
                latent=latent, d=1, D=D)


def generate_flat2(n, Q, D, sigma, a, replicate):
    from geometry.noise import sample_uniform_ball_noise
    emb = np.linalg.qr(_rng('flat2', 0, 7).standard_normal((D, 2)))[0]
    reference = _rng('flat2', replicate, 1).uniform(-1, 1, (n, 2)) @ emb.T
    latent = _rng('flat2', 0, 3).uniform(-.8, .8, (Q, 2)) @ emb.T
    g = _rng('flat2', 0, 4).standard_normal((Q, D))
    normals = _unit(g - (g @ emb) @ emb.T)
    noise = sample_uniform_ball_noise(n, D, sigma, _rng('flat2', replicate, 2))
    return dict(reference=reference + noise, clean_reference=reference, queries=latent + a * normals,
                latent=latent, d=2, D=D)


def generate_swiss3(n, Q, D, sigma, a, replicate):
    from geometry import candidate_swiss as cs
    from geometry.noise import sample_uniform_ball_noise
    def roll(count, rng, interior):
        # (theta, h) of the main Swiss roll (volume-uniform in arc length x height), plus a second flat direction
        pts, frames = cs.sample_native(count, rng, interior=interior)
        half = cs.HALF_HEIGHT - (cs.QUERY_BOUNDARY_MARGIN if interior else 0.)
        h2 = rng.uniform(-half, half, count)
        x = np.c_[pts, h2, np.zeros((count, D - 4))]
        tang = np.zeros((count, 3, D)); tang[:, :2, :3] = frames; tang[:, 2, 3] = 1.
        return x, tang
    reference, _ = roll(n, _rng('swiss3', replicate, 1), False)
    latent, T = roll(Q, _rng('swiss3', 0, 3), True)
    g = _rng('swiss3', 0, 4).standard_normal((Q, D))
    normals = _unit(g - np.einsum('qkd,qk->qd', T, np.einsum('qkd,qd->qk', T, g)))
    noise = sample_uniform_ball_noise(n, D, sigma, _rng('swiss3', replicate, 2))
    return dict(reference=reference + noise, clean_reference=reference, queries=latent + a * normals,
                latent=latent, d=3, D=D)


def _catenoid(count, rng, half):
    c, out = CATENOID_C, []
    while sum(map(len, out)) < count:  # v by rejection on the area element c cosh^2(v/c)
        v = rng.uniform(-half, half, max(512, count))
        out.append(v[rng.uniform(size=len(v)) < (np.cosh(v / c) / np.cosh(CATENOID_H / c)) ** 2])
    v = np.concatenate(out)[:count]; u = rng.uniform(0, 2 * np.pi, count)
    r = c * np.cosh(v / c)
    x = np.stack([r * np.cos(u), r * np.sin(u), v], axis=1)
    tu = np.stack([-np.sin(u), np.cos(u), 0 * u], axis=1)
    tv = _unit(np.stack([np.sinh(v / c) * np.cos(u), np.sinh(v / c) * np.sin(u), 1 + 0 * u], axis=1))
    return x, np.stack([tu, tv], axis=1)  # points (count, 3), orthonormal tangents (count, 2, 3)


def generate_catenoid(n, Q, D, sigma, a, replicate):
    from geometry.noise import sample_uniform_ball_noise
    emb = np.linalg.qr(_rng('catenoid', 0, 7).standard_normal((D, 3)))[0]
    reference = _catenoid(n, _rng('catenoid', replicate, 1), CATENOID_H)[0] @ emb.T
    latent, T = _catenoid(Q, _rng('catenoid', 0, 3), CATENOID_H - .1)
    latent, T = latent @ emb.T, T @ emb.T
    g = _rng('catenoid', 0, 4).standard_normal((Q, D))
    normals = _unit(g - np.einsum('qkd,qk->qd', T, np.einsum('qkd,qd->qk', T, g)))
    noise = sample_uniform_ball_noise(n, D, sigma, _rng('catenoid', replicate, 2))
    return dict(reference=reference + noise, clean_reference=reference, queries=latent + a * normals,
                latent=latent, d=2, D=D)


def central_noise(n, D, sigma, rng):
    direction = _unit(rng.standard_normal((n, D)))
    radius = sigma * (1 - np.sqrt(1 - rng.uniform(size=n)))  # inverse CDF of density 2(1-r/sigma)/sigma
    return direction * radius[:, None]


def make_data(spec):
    """spec: geometry, D, n, Q, sigma, a, noise, replicate."""
    g = spec['geometry']
    if g == 'fourier_torus':
        from experiments._shared.fourier_torus import sample_unit
        data = sample_unit(spec['n'], spec['Q'], spec['replicate'], sigma=spec['sigma'], a=spec['a'])
        data = dict(data, d=2, D=10)
    elif g == 'flat2':
        data = generate_flat2(spec['n'], spec['Q'], spec['D'], spec['sigma'], spec['a'], spec['replicate'])
    elif g == 'catenoid':
        data = generate_catenoid(spec['n'], spec['Q'], spec['D'], spec['sigma'], spec['a'], spec['replicate'])
    elif g == 'swiss3':
        data = generate_swiss3(spec['n'], spec['Q'], spec['D'], spec['sigma'], spec['a'], spec['replicate'])
    elif g in EXTRA:
        data = generate_extra(g, spec['n'], spec['Q'], spec['D'], spec['sigma'], spec['a'], spec['replicate'])
    else:
        data = generate(SyntheticConfig(geometry=g, n=spec['n'], Q=spec['Q'], D=spec['D'], sigma=spec['sigma'],
                                        a=spec['a'], replicate=spec['replicate'], seed=SEED))
    if spec.get('noise', 'uniform_ball') == 'central':
        if g in SEED_ORDER or g == 'fourier_torus':
            raise ValueError('central noise is defined for the synthetic main geometries only')
        from geometry.synthetic import _rng as synthetic_rng
        rng = synthetic_rng(SEED, g, spec['replicate'], 2)
        data = dict(data, reference=data['clean_reference'] + central_noise(spec['n'], data['D'], spec['sigma'], rng))
    return data
