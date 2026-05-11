"""
Render LALO20 barrier-certificate grid figures.

For each of the three instances LALO20/W001, W005, W01 we render a
2x3 grid of (x_i, x_4) projections in the 7-D state space:
  panels = (x_1, x_4), (x_2, x_4), (x_3, x_4),
           (x_5, x_4), (x_6, x_4), (x_7, x_4)

Each panel shows:
  * background heatmap of B(x) on the projection plane (other 5 state
    coordinates fixed at the initial-set midpoint),
  * green fill  = certified-safe sublevel   {B(x) <= gamma}     -> X_0,
  * red fill    = certified-unsafe superlevel {B(x) >= lambda}  -> X_u,
  * black-edged box = initial set X_0 projected onto the plane,
  * red box     = unsafe set X_u projected onto the plane,
  * grey streamlines = projected vector field f(x).

Reads `<RESULTS>/LALO20_<INST>.result.json` and writes
`<RESULTS>/LALO20_<INST>_grid.png` for INST in {W001, W005, W01}.
"""

import json
import os
import re

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get('PROTECT_RESULT_DIR', os.path.join(HERE, 'results'))


SAFE_FILL = '#7ed47e'
UNSAFE_FILL_LIGHT = '#f5a89c'
INIT_EDGE = '#000000'
INIT_FILL = '#ffffff'
UNSAFE_EDGE = '#7a0000'
UNSAFE_FILL = '#e74c3c'
STREAM_COLOR = '#444444'


def _sympify(s):
    s = re.sub(r'(\d)\.e([+-]?\d+)', r'\1.0e\2', s)
    return sp.sympify(s)


def _lalo20_dyn(s):
    x1, x2, x3, x4, x5, x6, x7 = s
    return [1.4 * x3 - 0.9 * x1,
            2.5 * x5 - 1.5 * x2,
            0.6 * x7 - 0.8 * x2 * x3,
            2.0 - 1.3 * x3 * x4,
            0.7 * x1 - x4 * x5,
            0.3 * x1 - 3.1 * x6,
            1.8 * x6 - 1.5 * x2 * x7]


def _draw_panel(ax, syms, barrier, gamma, lam, dim_x, dim_y, init_mid,
                L_init, U_init, unsafe_boxes, xlim, ylim, var_names, title):
    n = len(syms)
    other_subs = {syms[i]: float(init_mid[i])
                  for i in range(n) if i not in (dim_x, dim_y)}
    sliced_B = barrier.subs(other_subs)
    B_eval = sp.lambdify((syms[dim_x], syms[dim_y]), sliced_B, 'numpy')

    dyn = _lalo20_dyn(syms)
    fx = sp.lambdify((syms[dim_x], syms[dim_y]),
                     sp.sympify(dyn[dim_x]).subs(other_subs), 'numpy')
    fy = sp.lambdify((syms[dim_x], syms[dim_y]),
                     sp.sympify(dyn[dim_y]).subs(other_subs), 'numpy')

    nx = ny = 240
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)
    Zb = np.asarray(B_eval(X, Y), dtype=float)

    nxs = 24
    Xs, Ys = np.meshgrid(np.linspace(xlim[0], xlim[1], nxs),
                        np.linspace(ylim[0], ylim[1], nxs))
    FX = np.asarray(fx(Xs, Ys), dtype=float)
    FY = np.asarray(fy(Xs, Ys), dtype=float)
    if FX.ndim == 0: FX = np.full_like(Xs, float(FX))
    if FY.ndim == 0: FY = np.full_like(Xs, float(FY))

    im = ax.imshow(Zb, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   origin='lower', cmap='viridis', aspect='auto',
                   alpha=0.85, zorder=0)
    try:
        ax.streamplot(Xs, Ys, FX, FY, color=STREAM_COLOR, density=0.9,
                      linewidth=0.55, arrowsize=0.65, zorder=2)
    except Exception:
        pass

    if (Zb <= gamma).any():
        ax.contourf(X, Y, Zb, levels=[Zb.min() - 1, gamma],
                    colors=[SAFE_FILL], alpha=0.6, zorder=3)
    if (Zb >= lam).any():
        ax.contourf(X, Y, Zb, levels=[lam, Zb.max() + 1],
                    colors=[UNSAFE_FILL_LIGHT], alpha=0.45, zorder=3)

    for L_u, U_u in unsafe_boxes:
        L_u = np.asarray(L_u, float); U_u = np.asarray(U_u, float)
        slice_in_box = all(L_u[i] <= init_mid[i] <= U_u[i]
                           for i in range(n) if i not in (dim_x, dim_y))
        if not slice_in_box:
            continue
        wu = U_u[dim_x] - L_u[dim_x]; hu = U_u[dim_y] - L_u[dim_y]
        if wu <= 0 or hu <= 0:
            continue
        ax.add_patch(Rectangle((L_u[dim_x], L_u[dim_y]), wu, hu,
                               edgecolor=UNSAFE_EDGE, facecolor=UNSAFE_FILL,
                               alpha=0.85, linewidth=1.4, zorder=5))

    L0i = L_init[dim_x]; U0i = U_init[dim_x]
    L0j = L_init[dim_y]; U0j = U_init[dim_y]
    w0 = max(U0i - L0i, 0.015 * (xlim[1] - xlim[0]))
    h0 = max(U0j - L0j, 0.015 * (ylim[1] - ylim[0]))
    x0 = L0i if U0i > L0i else L0i - 0.5 * w0
    y0 = L0j if U0j > L0j else L0j - 0.5 * h0
    ax.add_patch(Rectangle((x0, y0), w0, h0,
                           edgecolor=INIT_EDGE, facecolor=INIT_FILL,
                           linewidth=1.4, zorder=6))

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(f'${var_names[dim_x]}$', fontsize=10)
    ax.set_ylabel(f'${var_names[dim_y]}$', fontsize=10)
    ax.set_title(title, fontsize=10)
    return im


def render(inst):
    inst_map = {'W001': (0.01, 4.5), 'W005': (0.05, 4.5), 'W01': (0.10, 5.0)}
    if inst not in inst_map:
        return
    W, x4_unsafe = inst_map[inst]
    label = f'LALO20_{inst}'
    rj = os.path.join(RESULTS, label + '.result.json')
    if not os.path.isfile(rj):
        print(f'  MISS {label}')
        return
    d = json.load(open(rj))
    if 'barrier' not in d or d.get('gamma') is None:
        print(f'  SKIP {label} (no barrier)')
        return
    barrier = _sympify(d['barrier'])
    gamma = float(d['gamma']); lam = float(d['lambda'])

    syms = sp.symbols('x0:7')
    centre = np.array([1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45])
    L_init = centre - W; U_init = centre + W
    init_mid = centre.copy()
    L_space = np.array([0.5, 0.5, 1.0, 1.5, 0.5, 0.05, 0.2])
    U_space = np.array([2.5, 2.5, 4.0, 6.0, 2.5, 0.5,  1.2])
    L_unsafe_v = L_space.copy(); L_unsafe_v[3] = x4_unsafe
    unsafe_boxes = [(L_unsafe_v, U_space)]
    var_names = [f'x_{i+1}' for i in range(7)]
    pairs = [(0, 3), (1, 3), (2, 3), (4, 3), (5, 3), (6, 3)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    last_im = None
    for ax, (dx, dy) in zip(axes.flat, pairs):
        xlim = (L_space[dx], U_space[dx])
        ylim = (L_space[dy], U_space[dy])
        last_im = _draw_panel(ax, syms, barrier, gamma, lam,
                              dx, dy, init_mid, L_init, U_init,
                              unsafe_boxes, xlim, ylim, var_names,
                              title=f'$({var_names[dx]},\\,{var_names[dy]})$ projection')

    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.78])
    fig.colorbar(last_im, cax=cbar_ax, label='$B(x)$')

    proxies = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SAFE_FILL, alpha=0.6,
                      label=rf'$\{{B \leq \gamma\}}$ ($\gamma = {gamma:.4g}$)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL_LIGHT, alpha=0.45,
                      label=rf'$\{{B \geq \lambda\}}$ ($\lambda = {lam:.4g}$)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=INIT_FILL, edgecolor=INIT_EDGE,
                      linewidth=1.4, label=r'initial set $X_0$'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL, edgecolor=UNSAFE_EDGE,
                      alpha=0.85, label=r'unsafe set $X_u$'),
        plt.Line2D([0], [0], color=STREAM_COLOR, lw=1.0, label=r'flow $f(x)$'),
    ]
    fig.legend(handles=proxies, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.95, bbox_to_anchor=(0.45, -0.005))
    fig.suptitle(f'{label}: barrier $B(x_1, \\dots, x_7)$ across $(x_i, x_4)$ '
                 f'projections  '
                 f'($\\gamma = {gamma:.4g}$, $\\lambda = {lam:.4g}$, '
                 f'$X_u = \\{{x_4 \\geq {x4_unsafe}\\}}$)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 0.91, 0.94])

    out = os.path.join(RESULTS, label + '_grid.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  ok  {out}')


def main():
    print('LALO20 grid figures:')
    for inst in ['W001', 'W005', 'W01']:
        render(inst)


if __name__ == '__main__':
    main()
