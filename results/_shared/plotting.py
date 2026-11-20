"""Manuscript figures and tables for the final campaign (2026-11-08), in the style of SIMREV paper_v2.py.

Statistic everywhere: mean over the reference datasets of the per-dataset latent RMSE, standard deviation across
datasets in parentheses (figures: +-1 sd bars). Writes <out>/{figures,tables,preview}.

usage: python -m results._shared.plotting <final_runs> <lowd_runs_m1> <case4_d10_csv> <out>

100-reference-dataset version (reps100, 2026-11-16): a copy of final_paper_assets.py in which only the cell
selection changes; every plotting and table-format line is the original. Default epsilon is 1, except the linked
tori (Case 5) and Case 6 at epsilon=2. LEGACY=1 restores the frozen selection (epsilon=2, 20 datasets, epsilon=2
Figure 1) and reproduces the frozen assets.
"""
import os
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments._shared.simulation_units import AMBIENT, DEFAULT_N, NGRID

BLUE, ORANGE, AQUA, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df'
MAIN = ('trefoil', 'torus', 'swiss_roll_candidate', 'bowl4')
TABLE_GEOMS = MAIN + ('double_graph',)
EXTRA = ('circle', 'flat2', 'catenoid')
XD, XN = dict(circle=5, flat2=5, catenoid=5), dict(circle=2000, flat2=8000, catenoid=8000)  # D and n per further geometry
LABEL = dict(trefoil='Trefoil', torus='Torus', swiss_roll_candidate='Swiss roll', double_graph='Double-curvature graph',
             bowl4='Bowl', linked_tori='Linked tori', fourier_torus='Fourier torus', circle='Circle',
             affine='Affine segment', figure_eight='Figure-8', helix='Helix', flat2='Plane', swiss3='Swiss roll', catenoid='Catenoid')
DIMS = dict(trefoil=(1, 5), torus=(2, 5), swiss_roll_candidate=(2, 5), double_graph=(3, 10), bowl4=(3, 10),
            linked_tori=(2, 5), fourier_torus=(2, 10), circle=(1, 5), affine=(1, 5), figure_eight=(1, 5), helix=(1, 5), flat2=(2, 5), swiss3=(3, 10), catenoid=(2, 5))
TEXLABEL = {g: f'{LABEL[g]} $({d},{D})$' for g, (d, D) in DIMS.items()}
LABEL['double_graph'] = 'Double graph'  # panel titles; tables keep the full name
METHODS = ('RAW', 'NP-MD-H', 'NP-MD-S', 'DP-MD-H', 'DP-MD-S')
PCA = 'DP-PCA-split-q0.5-Ratio'
EPSILONS = (1., 2., 4., 8.)
TOP = .10  # Case 4 figure: y axis cut just above RAW (.08)
MODES = dict(structured_shared='Joint', structured_independent='Per-scale', structured_per_query='Per-query')
EPS_LABEL = r'$\varepsilon$'
LEGACY = os.environ.get('LEGACY') == '1'
REPS = 20 if LEGACY else 100


def default_epsilon(g):
    return 2. if LEGACY or g == 'linked_tori' else 1.

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                     'lines.linewidth': 1.5, 'lines.markersize': 4})


def load_main(root):
    groups = defaultdict(lambda: defaultdict(list)); names = {}
    for p in sorted(Path(root, 'main', 'units').glob('*.json')):
        r = json.loads(p.read_text()); c = r['cell']
        k = (c['geometry'], c['D'], c['n'], c['epsilon'], c['a'], c['sigma'], c['noise'])
        for m, v in r['rmse'].items(): groups[k][m].append(v)
        names[k + (c['replicate'],)] = p.with_suffix('.npz')
    return groups, names


def key(g, *, D=None, n=None, epsilon=None, a=.08, sigma=.04, noise='uniform_ball'):
    epsilon = default_epsilon(g) if epsilon is None else epsilon
    return (g, D or AMBIENT[g], n or DEFAULT_N.get(g, 2000), float(epsilon), a, sigma, noise)


def stat(values):
    v = np.asarray(values, float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float('nan'), len(v)) if len(v) else None


def fmt(s, raw=False):
    if s is None: return '---'
    return f'${s[0]:.3f}$' if raw or not np.isfinite(s[1]) else f'${s[0]:.3f}$ \\scriptsize({s[1]:.3f})'


def write_table(path, header, rows):
    lines = [r'\begin{tabular}{l' + 'c' * (len(header) - 1) + '}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    for gi, (g, entries) in enumerate(rows):
        if gi: lines.append(r'\midrule')
        for ei, cells in enumerate(entries):
            lines.append((TEXLABEL.get(g, g) if ei == 0 else '') + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    path.write_text('\n'.join(lines) + '\n')


def method_cells(cell, methods=METHODS):
    return [fmt(stat(cell.get(m, [])), raw=m == 'RAW') for m in methods]


def tables(groups, TAB):
    hdr = lambda first, methods=METHODS: ['Geometry', first, *methods]
    write_table(TAB / 'tab_sim_hard_soft_a.tex', hdr('$a$'),
                [(g, [[f'${a:.2f}$', *method_cells(groups[key(g, a=a)])] for a in (.08, .10, .12, .14)]) for g in MAIN])
    write_table(TAB / 'tab_app_sweep_eps.tex', hdr(r'$\varepsilon$'),
                [(g, [[f'${e:g}$', *method_cells(groups[key(g, epsilon=e)])] for e in EPSILONS]) for g in TABLE_GEOMS])
    write_table(TAB / 'tab_app_sweep_n.tex', hdr('$n$'),
                [(g, [[f'${n}$', *method_cells(groups[key(g, n=n)])] for n in NGRID[g]]) for g in TABLE_GEOMS])
    write_table(TAB / 'tab_app_sweep_sigma.tex', hdr(r'$\sigma$'),
                [(g, [[f'${s:g}$', *method_cells(groups[key(g, sigma=s)])] for s in (.02, .04, .06, .08)]) for g in TABLE_GEOMS])
    write_table(TAB / 'tab_app_noise.tex', hdr('noise'),
                [(g, [[lab, *method_cells(groups[key(g, noise=nz)])] for nz, lab in (('uniform_ball', 'uniform'), ('central', 'central'))])
                 for g in TABLE_GEOMS])
    write_table(TAB / 'tab_app_hard_soft_a_dg.tex', hdr('$a$'),
                [('double_graph', [[f'${a:.2f}$', *method_cells(groups[key('double_graph', a=a)])] for a in (.08, .10, .12, .14)])])
    write_table(TAB / 'tab_app_extra.tex', hdr('$n$'), [(g, [[f'${XN[g]}$', *method_cells(groups[key(g, D=XD[g], n=XN[g])])]]) for g in EXTRA])
    soft = ('RAW', 'NP-MD-S', 'DP-MD-S')
    for g, name in (('linked_tori', 'tab_app_disconnected.tex'), ('fourier_torus', 'tab_app_fourier.tex')):
        write_table(TAB / name, ['Geometry', 'setting', *soft, 'DP-PCA'],
                    [(g, [[f'$n={n}$', *method_cells(groups[key(g, n=n)], soft + (PCA,))] for n in NGRID[g]] +
                      [[f'$\\varepsilon={e:g}$', *method_cells(groups[key(g, epsilon=e)], soft + (PCA,))] for e in EPSILONS if e != default_epsilon(g)])])
    # Case 6: the proposed estimators only (user 2026-11-08)
    # D varies within the table, so the row label is the geometry name with its sample size only
    write_table(TAB / 'tab_app_highd.tex', ['Geometry', '$D$', *METHODS],
                [(f'{name} ($n={n}$)', [[f'${D}$', *method_cells(groups[key(g, D=D, n=n, epsilon=2.)])] for D in (10, 20, 30, 40, 50)])
                 for g, n, name in (('trefoil', 8000, 'Trefoil'), ('double_graph', 16000, 'Double-curvature graph'))])


def manifold_table(names, TAB):
    """Case 3 in manifold RMSE: per-dataset root mean squared distance of the outputs to the true manifold."""
    from results._shared.manifold_distance import distance
    rows = []
    for g in TABLE_GEOMS:
        entries = []
        for a in (.08, .10, .12, .14):
            k = key(g, a=a); per = defaultdict(list)
            for r in range(REPS):
                with np.load(names[k + (r,)]) as z:
                    for m in METHODS:
                        per[m].append(float(np.sqrt(np.mean(distance(g, k[1], z['estimate_' + m]) ** 2))))
            entries.append([f'${a:.2f}$', *method_cells(per)])
        rows.append((g, entries))
    write_table(TAB / 'tab_app_hard_soft_a_manifold.tex', ['Geometry', '$a$', *METHODS], rows)


def comparator_tables(groups, root, m1_root, TAB):
    comp = defaultdict(lambda: defaultdict(dict))
    for folder in (Path(root, 'comp'), Path(m1_root)):
        for p in Path(folder, 'units').glob('*.json'):
            r = json.loads(p.read_text())
            if r['group'] not in ('sa', 'md', 'pca'): continue
            for k, v in r['rmse'].items(): comp[(r['geometry'], r['m'], r['epsilon'])][k].setdefault(r['replicate'], v)
    cols = ('DP-MD-S', 'NP-MD-S', PCA, 'SA-MF2023-B50', 'SA-MF2025-B50')
    names = ('DP-MD-S', 'NP-MD-S', 'DP-PCA', 'SA-MF2023', 'SA-MF2025')
    for m, name in ((1, 'tab_app_baselines.tex'), (20, 'tab_app_baselines_m20.tex')):
        rows = []
        for g in TABLE_GEOMS:
            entries = []
            for e in EPSILONS:
                c = comp[(g, m, e)]
                if m == 20:  # MD and PCA of the main campaign
                    c = dict(c, **{k: dict(enumerate(groups[key(g, epsilon=e)][k])) for k in ('DP-MD-S', 'NP-MD-S', PCA)})
                entries.append([f'${e:g}$', *[fmt(stat(list(c.get(k, {}).values()))) for k in cols]])
            rows.append((g, entries))
        write_table(TAB / name, ['Geometry', r'$\varepsilon$', *names], rows)


def case4(root, d10_csv):
    c4 = defaultdict(list)
    for p in Path(root, 'case4').glob('*.json'):
        r = json.loads(p.read_text()); c4[(r['geometry'], r['Q'], r['backend'])].append(r['rmse']['DP-MD-S'])
    with open(d10_csv) as f:
        for r in csv.DictReader(f):
            if r['geometry'] == 'double_graph': c4[('double_graph', int(r['m']), r['backend'])].append(float(r['DP-MD-S']))
    return c4


def case4_table(c4, TAB):
    write_table(TAB / 'tab_app_release_m.tex', ['Geometry', '$m$', *MODES.values()],
                [(g, [[f'${m}$', *[fmt(stat(c4[(g, m, b)])) for b in MODES]] for m in (10, 20, 50, 100)])
                 for g in ('trefoil', 'torus', 'double_graph')])


def _style(ax, title):
    ax.set_title(title, loc='left', color=INK)
    ax.grid(True, axis='y', color=GRID, linewidth=.6)
    ax.set_axisbelow(True)


def _line(ax, xs, cells, method, color, marker):
    v = [stat(c.get(method, [])) for c in cells]
    keep = [i for i, s in enumerate(v) if s]
    ax.errorbar([xs[i] for i in keep], [v[i][0] for i in keep], yerr=[v[i][1] for i in keep], color=color,
                marker=marker, capsize=2, elinewidth=.8, label=method)


def _save(fig, FIG, name):
    fig.savefig(FIG / name)
    fig.savefig(FIG.parent / 'preview' / name.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)


def sweep_figure(groups, FIG, which, grid_of, xlabel, path, geometries=MAIN, square=True):
    shape, size = ((2, 2), (3.5, 3.2)) if square else ((1, len(geometries)), (1.75 * len(geometries) + .3, 1.9))
    fig, axes = plt.subplots(*shape, figsize=size, squeeze=False)
    for ax, g in zip(axes.flat, geometries):
        grid = grid_of(g)
        cells = [groups[key(g, **{which: x})] for x in grid]
        _line(ax, grid, cells, 'DP-MD-S', BLUE, 'o')
        _line(ax, grid, cells, 'NP-MD-S', ORANGE, 's')
        ax.axhline(0.08, color=MUTED, linestyle=':', linewidth=1, label='RAW')
        if which != 'sigma': ax.set_xscale('log')  # sigma is an evenly spaced grid
        ax.set_xticks(grid); ax.minorticks_off()
        ax.set_xticklabels([f'{x // 1000:g}k' if which == 'n' else f'{x:g}' for x in grid])
        ax.set_ylim(0, 0.09)
        ax.set_xlabel(xlabel if not square or ax in axes[-1] else '')
        _style(ax, LABEL[g])
    for row in axes: row[0].set_ylabel('Latent RMSE')
    axes[0, 0].legend(frameon=False, loc='upper right', bbox_to_anchor=(1, .86))
    fig.tight_layout(w_pad=.6, h_pad=.4)
    _save(fig, FIG, path)


def two_panel_figure(groups, FIG, g, path):
    """Case 5 / Case 7 (as the disconnected figure of paper_v2): latent RMSE versus n and versus epsilon."""
    fig = plt.figure(figsize=(4.8, 2.0))
    for i, (which, grid, xl) in enumerate((('n', list(NGRID[g]), '$n$'), ('epsilon', list(EPSILONS), EPS_LABEL))):
        ax = fig.add_subplot(1, 2, i + 1)
        cells = [groups[key(g, **{which: x})] for x in grid]
        _line(ax, grid, cells, 'DP-MD-S', BLUE, 'o')
        _line(ax, grid, cells, 'NP-MD-S', ORANGE, 's')
        ax.axhline(0.08, color=MUTED, linestyle=':', linewidth=1, label='RAW')
        ax.set_xscale('log'); ax.set_xticks(grid); ax.minorticks_off()
        ax.set_xticklabels([f'{x // 1000:g}k' if which == 'n' else f'{x:g}' for x in grid])
        ax.set_ylim(0, 0.09); ax.set_xlabel(xl); _style(ax, '')
        if i == 0:
            ax.set_ylabel('Latent RMSE'); ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1, .86))
    fig.tight_layout()
    _save(fig, FIG, path)


def highd_figure(groups, FIG):
    """Case 6: trefoil and double-curvature graph against the ambient dimension (proposed estimators only)."""
    Ds = [10, 20, 30, 40, 50]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0), sharey=True)
    for ax, (g, n) in zip(axes, (('trefoil', 8000), ('double_graph', 16000))):
        cells = [groups[key(g, D=D, n=n, epsilon=2.)] for D in Ds]
        _line(ax, Ds, cells, 'DP-MD-S', BLUE, 'o')
        _line(ax, Ds, cells, 'NP-MD-S', ORANGE, 's')
        ax.axhline(0.08, color=MUTED, linestyle=':', linewidth=1, label='RAW')
        ax.set_xticks(Ds); ax.set_ylim(0, 0.09); ax.set_xlabel('$D$'); _style(ax, LABEL[g])
    axes[0].set_ylabel('Latent RMSE'); axes[0].legend(frameon=False, loc='upper right', bbox_to_anchor=(1, .86))
    fig.tight_layout()
    _save(fig, FIG, 'sim_highd.pdf')


def release_m_figure(c4, FIG, name='sim_release_m_grid.pdf'):
    """Case 4 (2026-11-17): DP-MD-S of each release against m, lines over layered ribbons, one panel per geometry,
    with ylim (0, TOP); a point above TOP is drawn on the top edge as an open marker with its value above it."""
    ms = [10, 20, 50, 100]; x = np.arange(len(ms))
    colors = dict(structured_shared='#1d4e89', structured_independent='#2a9d8f', structured_per_query='#c8553d')
    geoms = ('trefoil', 'torus', 'double_graph')
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1), sharey=True)  # the paper's size
    for ax, g in zip(axes, geoms):
        below = np.zeros(len(ms))
        for b, marker in zip(MODES, 'o^s'):
            y = np.array([np.mean(c4[(g, m, b)]) if c4[(g, m, b)] else np.nan for m in ms])
            yc = np.minimum(y, TOP)
            ax.fill_between(x, below, yc, color=colors[b], alpha=.16, linewidth=0)
            below = np.nan_to_num(yc)
            ax.plot(x, yc, color=colors[b], label=MODES[b], zorder=3)
            inside = y <= TOP
            ax.plot(x[inside], y[inside], linestyle='none', color=colors[b], marker=marker, zorder=4)
            ax.plot(x[~inside], yc[~inside], linestyle='none', marker=marker, markerfacecolor='white',
                    markeredgecolor=colors[b], zorder=4, clip_on=False)
            for xi, v in zip(x[~inside], y[~inside]):
                ax.annotate(f'{v:.2f}', (xi, TOP), xytext=(0, 5), textcoords='offset points', ha='center',
                            va='bottom', fontsize=6.5, color=colors[b], annotation_clip=False)
        ax.axhline(0.08, color=MUTED, linestyle=':', linewidth=1, label='RAW', zorder=2)
        ax.set_xticks(x); ax.set_xticklabels([str(m) for m in ms]); ax.set_xlim(-.4, len(ms) - .6)
        ax.set_ylim(0, TOP); ax.set_xlabel('$m$'); _style(ax, LABEL[g])
        ax.set_title(LABEL[g], loc='left', color=INK, pad=13)  # room for the values above the top edge
    axes[0].set_ylabel('Latent RMSE')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colors[b], marker=mk, label=MODES[b]) for b, mk in zip(MODES, 'o^s')]
    handles.append(Line2D([], [], color=MUTED, linestyle=':', linewidth=1, label='RAW'))
    # the trefoil panel is filled up to the cut, so the legend goes to the free space under the per-query line of
    # the last panel; margins are set by hand (tight_layout ignores the values above the axes)
    axes[-1].legend(handles=handles, frameon=False, loc='lower right', bbox_to_anchor=(1, .3), ncol=2, columnspacing=.6, fontsize=6.5,
                    handlelength=1.2, handletextpad=.4, borderaxespad=.1, labelspacing=.25)
    fig.subplots_adjust(left=.085, right=.99, bottom=.21, top=.79, wspace=.1)
    _save(fig, FIG, name)


def _native_basis(g, D):
    """Orthonormal map native -> R^D used by the data generators (identity padding for linked tori)."""
    if g == 'linked_tori': return np.eye(D, 3)
    if g == 'swiss3': return np.eye(D, 2)  # the spiral plane
    if g in ('affine', 'figure_eight', 'helix', 'flat2', 'catenoid'):
        from experiments._shared.final_geometries import _rng
        native = 2 if g in ('figure_eight', 'flat2') else 3
        return np.linalg.qr(_rng(g, 0, 7).standard_normal((D, native)))[0]
    from geometry.synthetic import _rng
    native = dict(trefoil=3, circle=2)[g]
    return np.linalg.qr(_rng(20261105, g, 0, 7).standard_normal((D, native)))[0]


def _truth(g):
    t = np.linspace(0, 2 * np.pi, 600)
    if g == 'trefoil': return np.stack([np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)], 1) / 3
    if g == 'circle': return np.c_[np.cos(t), np.sin(t)]
    if g == 'figure_eight': return np.c_[np.sin(t), np.sin(t) * np.cos(t)]
    if g == 'swiss3':  # the spiral of the Swiss roll, as placed by the generator
        from geometry import candidate_swiss as cs
        s = np.linspace(0, cs.THETA_MAX, 600); r = cs.RADIUS_START + cs.RADIAL_SLOPE * s
        return np.c_[r * np.cos(s), r * np.sin(s)] - np.asarray(cs.SURFACE_AREA_CENTER[:2])
    if g == 'helix':
        s = np.linspace(-2 * np.pi, 2 * np.pi, 600); return np.c_[.5 * np.cos(s), .5 * np.sin(s), .4 * s / np.pi]
    return np.c_[np.linspace(-1, 1, 50), np.zeros((50, 2))]


def _swiss3_coordinates(x):
    """Three-dimensional Swiss roll: (spiral angle rescaled to [-1, 1], distance to the roll) of points of R^10. The
    roll is spiral x [-H, H]^2 in the first four coordinates; the distance combines the distance to the spiral in the
    first two coordinates, the excess of the two flat coordinates over H and the last six coordinates."""
    from geometry import candidate_swiss as cs
    s = np.linspace(0, cs.THETA_MAX, 20001); r = cs.RADIUS_START + cs.RADIAL_SLOPE * s
    spiral = np.c_[r * np.cos(s), r * np.sin(s)] - np.asarray(cs.SURFACE_AREA_CENTER[:2])
    d2 = np.sum((x[:, None, :2] - spiral[None]) ** 2, axis=2); i = d2.argmin(1)
    flat = np.maximum(np.abs(x[:, 2:4]) - cs.HALF_HEIGHT, 0)
    dist = np.sqrt(d2[np.arange(len(x)), i] + np.sum(flat ** 2, 1) + np.sum(x[:, 4:] ** 2, 1))
    return np.c_[2 * s[i] / cs.THETA_MAX - 1, dist]


def _catenoid_coordinates(y, B):
    """Catenoid: (height rescaled to [-1, 1], distance to the catenoid) of points of R^5. The catenoid is a surface of
    revolution in the embedded 3-space B, so the distance combines the distance to the catenary in the meridian
    half-plane and the distance to that 3-space."""
    from experiments._shared.final_geometries import CATENOID_C as c, CATENOID_H as h
    yn = y @ B; s = np.linspace(-h, h, 20001)
    d2 = (np.hypot(yn[:, :1], yn[:, 1:2]) - c * np.cosh(s / c)) ** 2 + (yn[:, 2:] - s) ** 2; i = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(y)), i] + np.sum((y - yn @ B.T) ** 2, 1))
    return np.c_[s[i] / h, dist]


def vis_names(root):
    """Further-geometry figure: 100 public queries of reference dataset 0 (runs/vis100, merged by vis100.py)."""
    return {key(g, D=XD[g], n=XN[g]) + (0,): Path(root, 'vis100', f'{g}_m100_r0.npz') for g in EXTRA}


def _catenoid_wireframe(rings=7, meridians=12):
    from experiments._shared.final_geometries import CATENOID_C as c, CATENOID_H as h
    u, s = np.linspace(0, 2 * np.pi, 80), np.linspace(-h, h, 40)
    out = [np.c_[c * np.cosh(v / c) * np.cos(u), c * np.cosh(v / c) * np.sin(u), v + 0 * u] for v in np.linspace(-h, h, rings)]
    return out + [np.c_[c * np.cosh(s / c) * np.cos(w), c * np.cosh(s / c) * np.sin(w), s] for w in np.linspace(0, 2 * np.pi, meridians, endpoint=False)]


def _square_wireframe(lines=9):
    s, c = np.linspace(-1, 1, 2), np.linspace(-1, 1, lines)
    return [np.c_[s, v + 0 * s, 0 * s] for v in c] + [np.c_[v + 0 * s, s, 0 * s] for v in c]


def _estimates(names, k, r=0):
    with np.load(names[k + (r,)]) as z:
        return {m: z['estimate_' + m] for m in ('RAW', 'DP-MD-S', 'NP-MD-S')}, z['latent']


def reconstruction_figure(root, names, FIG, g='trefoil'):
    """Figure 1: the dedicated m=100 trefoil unit (final_runs/fig1), else replicate 0 of the m=20 default cell."""
    fig1 = Path(root, 'fig1', 'units', f'trefoil_D5_n2000_e{2 if LEGACY else 1}_m100_r0.npz')
    if fig1.exists():
        with np.load(fig1) as z:
            est, latent = {m: z['estimate_' + m] for m in ('RAW', 'DP-MD-S', 'NP-MD-S')}, z['latent']
    else:
        est, latent = _estimates(names, key(g))
    B = _native_basis(g, AMBIENT[g])
    curve = _truth(g)
    fig = plt.figure(figsize=(7.2, 2.5))
    for i, (name, y) in enumerate(est.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.plot(*curve.T, color=INK, linewidth=.9)
        for p, q in zip(y @ B, latent @ B): ax.plot(*np.c_[p, q], color=MUTED, linewidth=.4)
        ax.scatter(*(y @ B).T, s=7, color=(MUTED, BLUE, ORANGE)[i], depthshade=False, label=name)
        ax.view_init(28, 35); ax.set_axis_off()
    fig.legend([a.collections[-1] for a in fig.axes], list(est), loc='lower center', ncol=3, frameon=False, markerscale=2)
    fig.subplots_adjust(left=0, right=1, bottom=.08, top=1, wspace=0)
    _save(fig, FIG, 'sim_reconstruction.pdf')
    return {m: float(np.sqrt(np.mean(np.sum((est[m] - latent) ** 2, 1)))) for m in est}


def disconnected_view(names, root, FIG):
    """Case 5: the two linked tori drawn by their core circles (with four parallels of each tube), the outputs coloured
    by component; the bottom row magnifies, in the same projection, the region where the two rings cross."""
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import proj3d
    from geometry.linked_tori import component_labels, MAJOR_RADIUS as R, MINOR_RADIUS as r, CENTER_OFFSET as C
    g = 'linked_tori'; est, latent = _estimates(names, key(g)); B = _native_basis(g, 5)
    comp, lat = component_labels(latent), latent @ B
    t = np.linspace(0, 2 * np.pi, 1500)

    def ring(k, phi=None):  # core circle (phi None) or the parallel of the tube at angle phi
        rad, h = (R, 0 * t) if phi is None else (R + r * np.cos(phi), r * np.sin(phi) + 0 * t)
        x = np.c_[rad * np.cos(t), rad * np.sin(t), h]
        if k == 1: x = x[:, [0, 2, 1]]
        x[:, 0] += -C if k == 0 else C
        return x
    parallels, colour, light = (0, np.pi / 2, np.pi, 3 * np.pi / 2), (BLUE, ORANGE), ('#a9c9ef', '#f5bfa8')
    fig = plt.figure(figsize=(7.2, 3.6))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.5, 1), left=.01, right=.99, bottom=.02, top=.94, wspace=.05, hspace=.04)
    window = None
    for i, (name, y) in enumerate(est.items()):
        rmse = np.sqrt(np.mean(np.sum((y - latent) ** 2, 1))); y = y @ B
        ax = fig.add_subplot(grid[0, i], projection='3d')
        for k in (0, 1):
            for phi in parallels: ax.plot(*ring(k, phi).T, color=light[k], linewidth=.5)
            ax.plot(*ring(k).T, color=colour[k], linewidth=1.6)
        for p, q in zip(y, lat): ax.plot(*np.c_[p, q], color=INK, linewidth=.5)
        for k in (0, 1): ax.scatter(*y[comp == k].T, s=14, color=colour[k], edgecolors='white', linewidths=.4, depthshade=False)
        ax.set_title(f'{name}: RMSE {rmse:.3f}', color=INK); ax.view_init(28, -62); ax.set_axis_off()
        ax.set_box_aspect((1.6, 1, 1), zoom=1.3)
        fig.canvas.draw(); M = ax.get_proj()
        screen = lambda x: np.c_[proj3d.proj_transform(x[:, 0], x[:, 1], x[:, 2], M)[:2]]
        L2, Y2 = screen(lat), screen(y)
        if window is None: window = L2[[10, 13, 19]].mean(0), .022  # around three queries where the rings cross
        mid, h = window
        box = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * h + mid
        ax.add_artist(Line2D(*box.T, transform=ax.transData, color=INK, linewidth=.7, linestyle='--'))
        zoom = fig.add_subplot(grid[1, i])
        for k in (0, 1):
            for phi in parallels: zoom.plot(*screen(ring(k, phi)).T, color=light[k], linewidth=.8)
            zoom.plot(*screen(ring(k)).T, color=colour[k], linewidth=2)
        for p, q in zip(Y2, L2): zoom.plot(*np.c_[p, q], color=INK, linewidth=.8)
        for k in (0, 1):
            zoom.scatter(*L2[comp == k].T, s=14, facecolors='white', edgecolors=colour[k], linewidths=.8, zorder=3)
            zoom.scatter(*Y2[comp == k].T, s=20, color=colour[k], edgecolors='white', linewidths=.5, zorder=4)
        zoom.set_xlim(mid[0] - h, mid[0] + h); zoom.set_ylim(mid[1] - h, mid[1] + h); zoom.set_aspect('equal')
        zoom.set_xticks([]); zoom.set_yticks([])
        for s in zoom.spines.values(): s.set_visible(True); s.set_color(MUTED)
    _save(fig, FIG, 'app_sim_disconnected_view.pdf')


def extra_geometry_figure(names, FIG):
    fig = plt.figure(figsize=(7.2, 2.1 * len(EXTRA)))
    for gi, g in enumerate(EXTRA):
        est, latent = _estimates(names, key(g, D=XD[g], n=XN[g])); B = _native_basis(g, XD[g]); three = g in ('helix', 'catenoid', 'flat2')
        for mi, (name, y) in enumerate(est.items()):
            ax = fig.add_subplot(len(EXTRA), 3, 3 * gi + mi + 1, **({'projection': '3d'} if three else {}))
            if g == 'affine':  # along-segment coordinate against the distance to the segment (magnified)
                yn, ln = y @ B, latent @ B
                show = np.c_[yn[:, 0], np.linalg.norm(y - np.outer(yn[:, 0], B[:, 0]), axis=1)]
                lat = np.c_[ln[:, 0], 0 * ln[:, 0]]
                ax.plot([-1, 1], [0, 0], color=INK, linewidth=.9); ax.set_ylim(-.01, .12)
            elif g == 'flat2':  # true shape: the plane and a fixed normal direction, light wireframe of the square
                B3 = np.linalg.qr(B, mode='complete')[0][:, :3]
                show, lat = y @ B3, latent @ B3
                for x in _square_wireframe(): ax.plot(*x.T, color=GRID, linewidth=.6)
            elif g == 'catenoid':  # true shape in the embedded 3-space, light wireframe of the catenoid
                show, lat = y @ B, latent @ B
                for x in _catenoid_wireframe(): ax.plot(*x.T, color=GRID, linewidth=.6)
            elif g == 'swiss3':  # position along the spiral (rescaled to [-1, 1]) against the distance to the roll
                show, lat = _swiss3_coordinates(y), _swiss3_coordinates(latent)
                ax.plot([-1, 1], [0, 0], color=INK, linewidth=.9); ax.set_ylim(-.01, .12)
            else:
                show, lat = y @ B, latent @ B
                ax.plot(*_truth(g).T, color=INK, linewidth=.9)
            for p, q in zip(show, lat): ax.plot(*np.c_[p, q], color=MUTED, linewidth=.4)
            ax.scatter(*show.T, s=5, color=(MUTED, BLUE, ORANGE)[mi], **({'depthshade': False} if three else {}))
            rmse = np.sqrt(np.mean(np.sum((y - latent) ** 2, 1)))
            ax.set_title(f'{LABEL[g]}, {name}: {rmse:.3f}', color=INK, fontsize=7)
            if three: ax.view_init(20, 35)
            if g == 'catenoid': ax.set_box_aspect((1, 1, .75), zoom=1.45)
            if g == 'flat2': ax.set_zlim(-.3, .3); ax.set_box_aspect((1, 1, .3), zoom=1.45)
            ax.set_axis_off()
            if not three: ax.set_aspect('auto' if g in ('affine', 'flat2', 'swiss3', 'catenoid') else 'equal')
    fig.tight_layout()
    _save(fig, FIG, 'app_sim_extra_geometry.pdf')


def main():
    root, m1_root, d10_csv, out = map(Path, sys.argv[1:5])
    FIG, TAB = out / 'figures', out / 'tables'
    for d in (FIG, TAB, out / 'preview'): d.mkdir(parents=True, exist_ok=True)
    groups, names = load_main(root)
    tables(groups, TAB)
    comparator_tables(groups, root, m1_root, TAB)
    manifold_table(names, TAB)
    c4 = case4(root, d10_csv)
    case4_table(c4, TAB)
    sweep_figure(groups, FIG, 'epsilon', lambda g: list(EPSILONS), EPS_LABEL, 'sim_rmse_eps_half.pdf')
    sweep_figure(groups, FIG, 'n', lambda g: list(NGRID[g]), '$n$', 'sim_rmse_n_half.pdf')
    sweep_figure(groups, FIG, 'sigma', lambda g: [.02, .04, .06, .08], r'noise radius $\sigma$', 'app_sim_sigma.pdf',
                 geometries=TABLE_GEOMS, square=False)
    two_panel_figure(groups, FIG, 'linked_tori', 'sim_disconnected.pdf')
    two_panel_figure(groups, FIG, 'fourier_torus', 'sim_fourier.pdf')
    highd_figure(groups, FIG)
    release_m_figure(c4, FIG)
    disconnected_view(names, root, FIG)
    extra_geometry_figure(vis_names(root), FIG)
    print(json.dumps(dict(reconstruction=reconstruction_figure(root, names, FIG),
                          case4_reps={f'{g} m{m} {MODES[b]}': len(v) for (g, m, b), v in sorted(c4.items()) if g != 'double_graph'}), indent=1))


if __name__ == '__main__':
    main()
