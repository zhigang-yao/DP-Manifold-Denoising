"""Overview of every simulation geometry (user 2026-11-18): the reference sample of reference dataset 0 at the
default n of each geometry (sigma=.04, uniform-ball noise), in the native coordinates of its generator. The bowl and
the double-curvature graph (d=3) are shown on the slice where the third base coordinate is within .1 of zero; the
Fourier torus by its first three coordinates. usage: python -m results._shared.geometry_gallery <out dir> (writes figures/ and preview/)"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from experiments._shared.final_geometries import make_data, _rng as extra_rng
from geometry.synthetic import _rng as syn_rng

BLUE, ORANGE, INK = '#2a78d6', '#eb6834', '#0b0b0b'
CELLS = [('trefoil', 5, 2000, 'Trefoil $(1,5)$'), ('torus', 5, 8000, 'Torus $(2,5)$'),
         ('swiss_roll_candidate', 5, 8000, 'Swiss roll $(2,5)$'), ('bowl4', 10, 16000, 'Bowl $(3,10)$\nslice $|u_3|<0.1$'),
         ('double_graph', 10, 16000, 'Double graph $(3,10)$\nslice $|u_3|<0.1$'),
         ('linked_tori', 5, 16000, 'Linked tori $(2,5)$'), ('fourier_torus', 10, 16000, 'Fourier torus $(2,10)$\nfirst three coordinates'),
         ('circle', 5, 2000, 'Circle $(1,5)$'), ('flat2', 5, 8000, 'Plane $(2,5)$'), ('catenoid', 5, 8000, 'Catenoid $(2,5)$')]


def native(g, x, D):
    if g in ('trefoil', 'torus', 'circle'):
        k = dict(trefoil=3, torus=3, circle=2)[g]
        return x @ np.linalg.qr(syn_rng(20261105, g, 0, 7).standard_normal((D, k)))[0]
    if g in ('flat2', 'catenoid'):
        B = np.linalg.qr(extra_rng(g, 0, 7).standard_normal((D, 2 if g == 'flat2' else 3)))[0]
        return x @ (np.linalg.qr(B, mode='complete')[0][:, :3] if g == 'flat2' else B)
    return x  # identity padding (Swiss roll, bowl, double graph, linked tori) or native (Fourier torus)


def main(out):
    for d in ('figures', 'preview'): Path(out, d).mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 3.4))
    for i, (g, D, n, title) in enumerate(CELLS):
        data = make_data(dict(geometry=g, D=D, n=n, Q=20, sigma=.04, a=.08, noise='uniform_ball', replicate=0))
        x = native(g, data['reference'], D)
        if g in ('bowl4', 'double_graph'):
            keep = np.abs(x[:, 2]) < .1; x = x[keep][:, [0, 1, 3]]
        else:
            x = x[:, :3] if x.shape[1] >= 3 else np.c_[x, 0 * x[:, 0]]
        x = x[np.random.default_rng(0).permutation(len(x))[:3000]]
        ax = fig.add_subplot(2, 5, i + 1, projection='3d')
        colour = BLUE
        if g == 'linked_tori':
            from geometry.linked_tori import component_labels
            colour = np.where(component_labels(data['clean_reference'][:, :3])[np.random.default_rng(0).permutation(n)[:3000]] == 0, BLUE, ORANGE)
        ax.scatter(*x.T, s=.6, c=colour, alpha=.6, depthshade=False, linewidths=0)
        if g == 'circle': ax.view_init(90, -90)
        elif g == 'swiss_roll_candidate': ax.view_init(65, 35)
        else: ax.view_init(20, 35)
        ax.set_box_aspect(np.ptp(x, axis=0) / np.ptp(x, axis=0).max() * (1 if g != 'flat2' else 1) + .05, zoom=1.25)
        ax.set_axis_off(); ax.set_title(title, fontsize=6.5, color=INK, pad=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=.94, wspace=0, hspace=.12)
    fig.savefig(Path(out, 'figures', 'app_sim_geometries.pdf'))
    fig.savefig(Path(out, 'preview', 'app_sim_geometries.png'), dpi=150)


if __name__ == '__main__':
    main(sys.argv[1])
