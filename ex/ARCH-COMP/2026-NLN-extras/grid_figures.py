"""
Multi-panel barrier-certificate figures.

Renders a grid of 2-D projections of B(x) for high-dimensional benchmarks:

  * CVDP23/b2, CVDP23/b_unc  ->  2x2 grid showing (x1, y1), (x2, y1),
                                  (x1, y2), (x2, y2) projections of the
                                  4-D state.
  * LALO20/W001, W005, W01    ->  2x3 grid showing (x_i, x_4) for
                                  i in {1, 2, 3, 5, 6, 7} -- the unsafe
                                  threshold is on x_4 so x_4 is the
                                  critical axis to keep on every panel.

Each panel shows the vector field as background streamlines, the initial
set (white-filled black-edged box), the unsafe set (red box, only when
the orthogonal slice values place the box inside the visible plane),
and the two barrier level sets B = gamma (green) and B = lambda (red).

When gamma ~ lambda the two contours become a lower/upper bracket pair
sandwiching the single certificate level, with a yellow fill in between.

Inputs:
    results/<label>.result.json (saved barrier + gamma + lambda).

Outputs:
    results/<label>_grid.png.
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
RESULTS = os.path.join(HERE, 'results')


# Layer colours.
GREEN_LINE  = '#1f7a1f'
RED_LINE    = '#a51d1d'
INIT_EDGE   = '#000000'
INIT_FILL   = '#ffffff'
UNSAFE_EDGE = '#7a0000'
UNSAFE_FILL = '#e74c3c'
STREAM_COLOR = '#666666'
TIED_BAND   = '#fde2a4'


def _sympify_barrier(s):
    s = re.sub(r'(\d)\.e([+-]?\d+)', r'\1.0e\2', s)
    return sp.sympify(s)


def _render_panel(ax, syms, barrier, gamma, lam, dyn_fn,
                  dim_x, dim_y, init_mid, L_init, U_init,
                  unsafe_boxes, xlim, ylim, var_names, title):
    """Render a single (dim_x, dim_y) projection on `ax`."""
    n = len(syms)
    other_subs = {syms[i]: float(init_mid[i])
                  for i in range(n) if i not in (dim_x, dim_y)}

    sliced_B = barrier.subs(other_subs)
    B_eval = sp.lambdify((syms[dim_x], syms[dim_y]), sliced_B, 'numpy')

    dyn = dyn_fn(syms)
    fx_expr = sp.sympify(dyn[dim_x]).subs(other_subs)
    fy_expr = sp.sympify(dyn[dim_y]).subs(other_subs)
    fx = sp.lambdify((syms[dim_x], syms[dim_y]), fx_expr, 'numpy')
    fy = sp.lambdify((syms[dim_x], syms[dim_y]), fy_expr, 'numpy')

    # Grids.
    nx = ny = 240
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)
    Zb = np.asarray(B_eval(X, Y), dtype=float)

    nxs = nys = 30
    Xs, Ys = np.meshgrid(np.linspace(xlim[0], xlim[1], nxs),
                        np.linspace(ylim[0], ylim[1], nys))
    FX = np.asarray(fx(Xs, Ys), dtype=float)
    FY = np.asarray(fy(Xs, Ys), dtype=float)
    if FX.ndim == 0: FX = np.full_like(Xs, float(FX))
    if FY.ndim == 0: FY = np.full_like(Xs, float(FY))

    # Layer 1: vector field.
    if np.isfinite(FX).any() and np.isfinite(FY).any():
        try:
            speed = np.hypot(FX, FY)
            speed_safe = np.where(speed > 0, speed, 1e-9)
            lw_min, lw_max = np.log10(speed_safe.min()), np.log10(speed_safe.max())
            lw = 0.5 + 0.7 * (np.log10(speed_safe) - lw_min) / max(lw_max - lw_min, 1e-6)
            ax.streamplot(Xs, Ys, FX, FY, color=STREAM_COLOR, density=0.95,
                          linewidth=lw, arrowsize=0.8, zorder=1)
        except Exception:
            ax.quiver(Xs, Ys, FX, FY, color=STREAM_COLOR, alpha=0.6, zorder=1)

    # Layer 3: unsafe boxes (only if slice is inside the box in n-D).
    for L_u, U_u in unsafe_boxes:
        L_u = np.asarray(L_u, float); U_u = np.asarray(U_u, float)
        slice_in_box = True
        for i in range(n):
            if i in (dim_x, dim_y):
                continue
            if not (L_u[i] <= init_mid[i] <= U_u[i]):
                slice_in_box = False
                break
        if not slice_in_box:
            continue
        wu = U_u[dim_x] - L_u[dim_x]
        hu = U_u[dim_y] - L_u[dim_y]
        if wu <= 0 or hu <= 0:
            continue
        ax.add_patch(Rectangle((L_u[dim_x], L_u[dim_y]), wu, hu,
                               edgecolor=UNSAFE_EDGE, facecolor=UNSAFE_FILL,
                               alpha=0.85, linewidth=1.5, zorder=3))

    # Layer 4: initial set (projected).
    L0i = L_init[dim_x]; U0i = U_init[dim_x]
    L0j = L_init[dim_y]; U0j = U_init[dim_y]
    w0 = max(U0i - L0i, 0.015 * (xlim[1] - xlim[0]))
    h0 = max(U0j - L0j, 0.015 * (ylim[1] - ylim[0]))
    x0 = L0i if U0i > L0i else L0i - 0.5 * w0
    y0 = L0j if U0j > L0j else L0j - 0.5 * h0
    ax.add_patch(Rectangle((x0, y0), w0, h0,
                           edgecolor=INIT_EDGE, facecolor=INIT_FILL,
                           linewidth=1.6, zorder=4))

    # Layer 6/8: level sets.
    Zmin, Zmax = float(np.nanmin(Zb)), float(np.nanmax(Zb))
    B_at_init = float(B_eval(0.5 * (L0i + U0i), 0.5 * (L0j + U0j)))
    rel = abs(lam - gamma) / max(abs(gamma), abs(lam), 1.0)
    tied = rel < 0.05
    if not tied:
        plot_gamma = max(gamma, B_at_init); plot_lambda = lam
        ax.contour(X, Y, Zb, levels=[plot_gamma],
                   colors=[GREEN_LINE], linewidths=2.0, zorder=8)
        ax.contour(X, Y, Zb, levels=[plot_lambda],
                   colors=[RED_LINE], linewidths=2.0, zorder=8)
    else:
        # Tied gamma ~ lambda case: draw a SINGLE dividing line at the
        # certificate level. Using a wide [gamma-eps, lambda+eps] band
        # filled with colour is misleading for B with a large value
        # range, because the fill ends up covering most of the slice
        # (every point where B is between Zmin and lo+eps). Just emit
        # the level set itself, with green/red double-stroke styling so
        # both bracket labels make sense in the legend.
        ax.contour(X, Y, Zb, levels=[B_at_init],
                   colors=[GREEN_LINE], linewidths=3.2, zorder=8)
        ax.contour(X, Y, Zb, levels=[B_at_init],
                   colors=[RED_LINE], linewidths=1.2,
                   linestyles='--', zorder=9)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(f'${var_names[dim_x]}$', fontsize=10)
    ax.set_ylabel(f'${var_names[dim_y]}$', fontsize=10)
    ax.set_title(title, fontsize=10)
    return tied


def _shared_legend(fig, tied, gamma, lam, vf_label):
    proxies = [
        plt.Line2D([0], [0], color=GREEN_LINE, lw=2.0,
                   label=(rf'$B(x) = \gamma \approx \lambda = {gamma:.4g}$ (lower bracket)'
                          if tied else rf'$B(x) = \gamma = {gamma:.4g}$')),
        plt.Line2D([0], [0], color=RED_LINE, lw=2.0,
                   linestyle=('--' if tied else '-'),
                   label=(r'upper bracket (same level)'
                          if tied else rf'$B(x) = \lambda = {lam:.4g}$')),
        plt.Line2D([0], [0], color=STREAM_COLOR, lw=1.2,
                   label=vf_label),
        plt.Rectangle((0, 0), 1, 1, facecolor=INIT_FILL,
                      edgecolor=INIT_EDGE, linewidth=1.5,
                      label=r'initial set $X_0$'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL,
                      edgecolor=UNSAFE_EDGE, alpha=0.85,
                      label=r'unsafe set $X_u$'),
    ]
    fig.legend(handles=proxies, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, -0.005))


# ---------------------------------------------------------------------
# Benchmark dynamics + specs.
# ---------------------------------------------------------------------

def cvdp_dyn(b):
    def f(s):
        x1, y1, x2, y2 = s
        return [y1,
                (1.0)*(1 - x1**2)*y1 + b*(x2 - x1) - x1,
                y2,
                (1.0)*(1 - x2**2)*y2 - b*(x2 - x1) - x2]
    return f


def lalo20_dyn(s):
    x1, x2, x3, x4, x5, x6, x7 = s
    return [
        1.4 * x3 - 0.9 * x1,
        2.5 * x5 - 1.5 * x2,
        0.6 * x7 - 0.8 * x2 * x3,
        2.0 - 1.3 * x3 * x4,
        0.7 * x1 - x4 * x5,
        0.3 * x1 - 3.1 * x6,
        1.8 * x6 - 1.5 * x2 * x7,
    ]


# ---------------------------------------------------------------------
# Per-benchmark drivers.
# ---------------------------------------------------------------------

def render_cvdp23(label, b_eff):
    """2x2 grid of (x1,y1), (x2,y1), (x1,y2), (x2,y2) projections."""
    rj = os.path.join(RESULTS, label + '.result.json')
    if not os.path.isfile(rj):
        return False
    with open(rj) as fp:
        data = json.load(fp)
    if 'barrier' not in data or data.get('gamma') is None:
        return False
    barrier = _sympify_barrier(data['barrier'])
    gamma = float(data['gamma']); lam = float(data['lambda'])

    syms = sp.symbols('x0:4')
    L_init = np.array([1.25, 2.35, 1.25, 2.35])
    U_init = np.array([1.55, 2.45, 1.55, 2.45])
    init_mid = 0.5 * (L_init + U_init)
    unsafe_boxes = [
        ([-3.0, 2.75, -3.0, -3.0], [3.0, 3.0, 3.0, 3.0]),
        ([-3.0, -3.0, -3.0, 2.75], [3.0, 3.0, 3.0, 3.0]),
    ]
    var_names = ['x_1', 'y_1', 'x_2', 'y_2']
    pairs = [(0, 1), (2, 1), (0, 3), (2, 3)]
    xlim = ylim = (-3.0, 3.0)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
    tied = False
    for ax, (dx, dy) in zip(axes.flat, pairs):
        t = _render_panel(ax, syms, barrier, gamma, lam, cvdp_dyn(b_eff),
                          dx, dy, init_mid, L_init, U_init,
                          unsafe_boxes, xlim, ylim, var_names,
                          title=f'$({var_names[dx]},\\,{var_names[dy]})$ projection')
        tied = tied or t

    _shared_legend(fig, tied, gamma, lam,
                   rf'vector field $f(x, b={b_eff:g})$')
    fig.suptitle(f'{label}: barrier $B(x_1, y_1, x_2, y_2)$ across all '
                 f'2-D projections  '
                 f'($\\gamma = {gamma:.4g}$, $\\lambda = {lam:.4g}$)',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path = os.path.join(RESULTS, label + '_grid.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  ok  {label}_grid.png  (gamma={gamma:.4g}, lambda={lam:.4g})')
    return True


def render_lalo20(label):
    """2x3 grid of (x_i, x_4) projections for i in {1, 2, 3, 5, 6, 7}.
    x_4 is the unsafe-threshold dimension so it stays on every panel."""
    rj = os.path.join(RESULTS, label + '.result.json')
    if not os.path.isfile(rj):
        return False
    with open(rj) as fp:
        data = json.load(fp)
    if 'barrier' not in data or data.get('gamma') is None:
        return False
    barrier = _sympify_barrier(data['barrier'])
    gamma = float(data['gamma']); lam = float(data['lambda'])

    # Instance parameters.
    inst_map = {'LALO20_W001': (0.01, 4.5),
                'LALO20_W005': (0.05, 4.5),
                'LALO20_W01':  (0.10, 5.0)}
    W, x4_unsafe = inst_map[label]

    syms = sp.symbols('x0:7')
    centre = np.array([1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45])
    L_init = centre - W
    U_init = centre + W
    init_mid = centre.copy()
    L_space = np.array([0.5, 0.5, 1.0, 1.5, 0.5, 0.05, 0.2])
    U_space = np.array([2.5, 2.5, 4.0, 6.0, 2.5, 0.5,  1.2])
    L_unsafe = L_space.copy(); L_unsafe[3] = x4_unsafe
    unsafe_boxes = [(L_unsafe, U_space)]
    var_names = [f'x_{i+1}' for i in range(7)]

    pairs = [(0, 3), (1, 3), (2, 3),
             (4, 3), (5, 3), (6, 3)]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    tied = False
    for ax, (dx, dy) in zip(axes.flat, pairs):
        xlim = (L_space[dx], U_space[dx])
        ylim = (L_space[dy], U_space[dy])
        t = _render_panel(ax, syms, barrier, gamma, lam, lalo20_dyn,
                          dx, dy, init_mid, L_init, U_init,
                          unsafe_boxes, xlim, ylim, var_names,
                          title=f'$({var_names[dx]},\\,{var_names[dy]})$ projection')
        tied = tied or t

    _shared_legend(fig, tied, gamma, lam, r'vector field $f(x)$')
    fig.suptitle(f'{label}: barrier $B(x_1, \\dots, x_7)$ across $(x_i, x_4)$ '
                 f'projections  '
                 f'($\\gamma = {gamma:.4g}$, $\\lambda = {lam:.4g}$, '
                 f'$X_u = \\{{x_4 \\geq {x4_unsafe}\\}}$)',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path = os.path.join(RESULTS, label + '_grid.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  ok  {label}_grid.png  (gamma={gamma:.4g}, lambda={lam:.4g})')
    return True


def main():
    print('CVDP23 grid figures:')
    render_cvdp23('CVDP23_b2',    b_eff=2.0)
    render_cvdp23('CVDP23_b_unc', b_eff=2.0)
    print()
    print('LALO20 grid figures:')
    for inst in ['LALO20_W001', 'LALO20_W005', 'LALO20_W01']:
        render_lalo20(inst)


if __name__ == '__main__':
    main()
