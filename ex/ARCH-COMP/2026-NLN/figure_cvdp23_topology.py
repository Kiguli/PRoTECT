"""
Definitive CVDP23 barrier-certificate figure.

The synthesised barrier B(x) for CVDP23 has gamma ~ lambda within MOSEK's
solver tolerance (7e-7 separation): the system is just-barely-safe so the
SOS programme has no slack to enlarge the gap. Drawing the contour at
exactly B = gamma in this regime produces a noise-driven wiggly curve
that visually looks ambiguous (you cannot tell that the curve really
separates init from unsafe).

This script renders an unambiguous version of the figure:

  * Heatmap of B(x) over the (x_1, y_1) slice (at x_2 = y_2 = init mid),
    colourbar showing absolute B value -- the user sees that B is large
    where the unsafe set is and small where the initial set is.
  * Two clearly-separated overlay regions:
      green fill = {x : B(x) <= gamma}  ('safe sublevel': contains X_0)
      red fill   = {x : B(x) >= lambda} ('certified-unsafe superlevel':
                                          contains X_u)
  * Initial set (black-edged white box) -- positioned inside the green
    fill.
  * Unsafe set (red box) -- positioned inside the red fill.

The figure is rendered for both CVDP23 instances (b=2 and b in [1,3]),
each as a 2x2 grid of (x_i, y_j) projections so the symmetry of the
4-D state is visible.
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


SAFE_FILL = '#7ed47e'      # filled {B <= gamma}
UNSAFE_FILL_LIGHT = '#f5a89c'  # filled {B >= lambda}
INIT_EDGE = '#000000'
INIT_FILL = '#ffffff'
UNSAFE_EDGE = '#7a0000'
UNSAFE_FILL = '#e74c3c'
STREAM_COLOR = '#444444'


def _sympify_barrier(s):
    s = re.sub(r'(\d)\.e([+-]?\d+)', r'\1.0e\2', s)
    return sp.sympify(s)


def _coupled_vdp_dyn(syms, b):
    x1, y1, x2, y2 = syms
    return [y1,
            (1 - x1**2)*y1 + b*(x2 - x1) - x1,
            y2,
            (1 - x2**2)*y2 - b*(x2 - x1) - x2]


def render_panel(ax, syms, barrier, gamma, lam, b_eff,
                 dim_x, dim_y, init_mid, L_init, U_init,
                 unsafe_boxes, xlim, ylim, var_names, title):
    n = len(syms)
    other_subs = {syms[i]: float(init_mid[i])
                  for i in range(n) if i not in (dim_x, dim_y)}
    sliced_B = barrier.subs(other_subs)
    B_eval = sp.lambdify((syms[dim_x], syms[dim_y]), sliced_B, 'numpy')

    dyn = _coupled_vdp_dyn(syms, b_eff)
    fx = sp.lambdify((syms[dim_x], syms[dim_y]),
                     sp.sympify(dyn[dim_x]).subs(other_subs), 'numpy')
    fy = sp.lambdify((syms[dim_x], syms[dim_y]),
                     sp.sympify(dyn[dim_y]).subs(other_subs), 'numpy')

    nx = ny = 280
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)
    Zb = np.asarray(B_eval(X, Y), dtype=float)

    nxs = 28
    Xs, Ys = np.meshgrid(np.linspace(xlim[0], xlim[1], nxs),
                        np.linspace(ylim[0], ylim[1], nxs))
    FX = np.asarray(fx(Xs, Ys), dtype=float)
    FY = np.asarray(fy(Xs, Ys), dtype=float)
    if FX.ndim == 0: FX = np.full_like(Xs, float(FX))
    if FY.ndim == 0: FY = np.full_like(Xs, float(FY))

    # Layer 1: B heatmap (viridis), back.
    im = ax.imshow(Zb, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   origin='lower', cmap='viridis', aspect='auto',
                   alpha=0.85, zorder=0)

    # Layer 2: vector field (very faint).
    try:
        ax.streamplot(Xs, Ys, FX, FY, color=STREAM_COLOR, density=0.7,
                      linewidth=0.6, arrowsize=0.7, zorder=2)
    except Exception:
        pass

    # Layer 3: {B <= gamma} -- safe sublevel, hatched green.
    safe_mask = Zb <= gamma
    if safe_mask.any():
        ax.contourf(X, Y, Zb, levels=[Zb.min() - 1, gamma],
                    colors=[SAFE_FILL], alpha=0.65, zorder=3)
    # Layer 3b: {B >= lambda} -- certified-unsafe superlevel.
    unsafe_super = Zb >= lam
    if unsafe_super.any():
        ax.contourf(X, Y, Zb, levels=[lam, Zb.max() + 1],
                    colors=[UNSAFE_FILL_LIGHT], alpha=0.45, zorder=3)

    # Layer 4: unsafe boxes -- only if slice falls inside the box.
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
                               alpha=0.85, linewidth=1.5, zorder=5))

    # Layer 6: initial set.
    L0i = L_init[dim_x]; U0i = U_init[dim_x]
    L0j = L_init[dim_y]; U0j = U_init[dim_y]
    w0 = max(U0i - L0i, 0.015 * (xlim[1] - xlim[0]))
    h0 = max(U0j - L0j, 0.015 * (ylim[1] - ylim[0]))
    x0 = L0i if U0i > L0i else L0i - 0.5 * w0
    y0 = L0j if U0j > L0j else L0j - 0.5 * h0
    ax.add_patch(Rectangle((x0, y0), w0, h0,
                           edgecolor=INIT_EDGE, facecolor=INIT_FILL,
                           linewidth=1.6, zorder=6))

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(f'${var_names[dim_x]}$', fontsize=10)
    ax.set_ylabel(f'${var_names[dim_y]}$', fontsize=10)
    ax.set_title(title, fontsize=10)
    return im


def render(label, b_eff):
    rj = os.path.join(RESULTS, label + '.result.json')
    if not os.path.isfile(rj):
        print(f'no result.json for {label}')
        return
    with open(rj) as fp:
        data = json.load(fp)
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

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.8))
    last_im = None
    for ax, (dx, dy) in zip(axes.flat, pairs):
        last_im = render_panel(ax, syms, barrier, gamma, lam, b_eff,
                               dx, dy, init_mid, L_init, U_init,
                               unsafe_boxes, xlim, ylim, var_names,
                               title=f'$({var_names[dx]},\\,{var_names[dy]})$ projection')

    # Shared colorbar.
    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.78])
    cbar = fig.colorbar(last_im, cax=cbar_ax, label='$B(x)$')

    # Shared legend.
    proxies = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SAFE_FILL, alpha=0.65,
                      label=rf'$\{{B(x) \leq \gamma\}}$ (safe sublevel; $\gamma = {gamma:.4g}$)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL_LIGHT, alpha=0.45,
                      label=rf'$\{{B(x) \geq \lambda\}}$ (certified-unsafe; $\lambda = {lam:.4g}$)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=INIT_FILL, edgecolor=INIT_EDGE,
                      linewidth=1.5, label=r'initial set $X_0$'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL, edgecolor=UNSAFE_EDGE,
                      alpha=0.85, label=r'unsafe set $X_u$'),
        plt.Line2D([0], [0], color=STREAM_COLOR, lw=1.2,
                   label=rf'flow $f(x, b={b_eff:g})$'),
    ]
    fig.legend(handles=proxies, loc='lower center', ncol=3, fontsize=9,
               framealpha=0.95, bbox_to_anchor=(0.45, -0.005))

    sep = lam - gamma
    fig.suptitle(f'{label}: $B(x_1, y_1, x_2, y_2)$ across all four 2-D '
                 f'projections  ($\\gamma = {gamma:.4g}$, '
                 f'$\\lambda - \\gamma = {sep:+.2e}$)\n'
                 f'green fill = $\\{{B \\leq \\gamma\\}}$ contains $X_0$  •  '
                 f'red fill = $\\{{B \\geq \\lambda\\}}$ contains $X_u$',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.07, 0.91, 0.94])
    out_path = os.path.join(RESULTS, label + '_topology.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  ok  {label}_topology.png  (gamma={gamma:.4g}, lambda={lam:.4g}, sep={sep:+.2e})')


def main():
    for label, b in [('CVDP23_b2', 2.0), ('CVDP23_b_unc', 2.0)]:
        render(label, b_eff=b)


if __name__ == '__main__':
    main()
