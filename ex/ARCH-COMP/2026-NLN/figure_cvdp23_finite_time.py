"""
Visualise the finite-time CVDP23 barrier B(x, t) at several t slices.

The finite-time solver produced a time-augmented certificate
    B(x, t) = B_0(x) + t * B_1(x) + t^2 * B_2(x)
for t in [0, T_horizon] (T=7 for the paper spec). Unlike the
infinite-horizon certificate -- which collapsed to gamma ~ lambda
because the dynamics' reach tube grazes the unsafe boundary -- the
finite-time version has time-dependent freedom and the level sets
{B(x, t) <= gamma} and {B(x, t) >= lambda} are visible at every t.

We produce a 2x3 grid:
  rows = projection plane (x1, y1) and (x2, y2)
  cols = t slice (t = 0, T/2, T)

Each panel shows:
  * heatmap of B(x, t) on the chosen slice (the other two state
    coords fixed at the init-set midpoint)
  * green fill = {B <= gamma}, red fill = {B >= lambda}
  * initial set X_0 (black-edged white box)
  * unsafe set X_u (red box)
  * vector field f(x, b=2) as streamlines
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


def _coupled_vdp_dyn(syms, b):
    x1, y1, x2, y2 = syms
    return [y1,
            (1 - x1**2) * y1 + b * (x2 - x1) - x1,
            y2,
            (1 - x2**2) * y2 - b * (x2 - x1) - x2]


def render_panel(ax, syms, barrier_t, gamma, lam, b_eff,
                 dim_x, dim_y, init_mid, L_init, U_init,
                 unsafe_boxes, xlim, ylim, var_names, title):
    n = len(syms)
    other_subs = {syms[i]: float(init_mid[i])
                  for i in range(n) if i not in (dim_x, dim_y)}
    sliced_B = barrier_t.subs(other_subs)
    B_eval = sp.lambdify((syms[dim_x], syms[dim_y]), sliced_B, 'numpy')

    dyn = _coupled_vdp_dyn(syms, b_eff)
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
        ax.streamplot(Xs, Ys, FX, FY, color=STREAM_COLOR, density=0.6,
                      linewidth=0.5, arrowsize=0.6, zorder=2)
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


def main():
    rj = os.path.join(RESULTS, 'CVDP23_finite_time.result.json')
    if not os.path.isfile(rj):
        print('CVDP23_finite_time.result.json not found')
        return
    with open(rj) as fp:
        data = json.load(fp)
    barrier_full = _sympify(data['barrier'])
    gamma = float(data['gamma']); lam = float(data['lambda'])
    T_horizon = float(data.get('T_horizon', 7.0))

    syms = sp.symbols('x0:4')
    # Time symbol used by the solver -- find it among the free symbols.
    free_extra = [s for s in barrier_full.free_symbols if s not in syms]
    if not free_extra:
        print('barrier does not depend on time -- treating as t=0 only')
        t_sym = None
    else:
        t_sym = free_extra[0]
        print(f'time symbol detected: {t_sym}')

    L_init = np.array([1.25, 2.35, 1.25, 2.35])
    U_init = np.array([1.55, 2.45, 1.55, 2.45])
    init_mid = 0.5 * (L_init + U_init)
    unsafe_boxes = [
        ([-3.0, 2.75, -3.0, -3.0], [3.0, 3.0, 3.0, 3.0]),
        ([-3.0, -3.0, -3.0, 2.75], [3.0, 3.0, 3.0, 3.0]),
    ]
    var_names = ['x_1', 'y_1', 'x_2', 'y_2']
    proj_rows = [(0, 1, '(x_1, y_1)'), (2, 3, '(x_2, y_2)')]
    t_cols = [0.0, T_horizon / 2.0, T_horizon]
    xlim = ylim = (-3.0, 3.0)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0))
    last_im = None
    for row_i, (dx, dy, lbl) in enumerate(proj_rows):
        for col_i, t_val in enumerate(t_cols):
            ax = axes[row_i, col_i]
            if t_sym is None:
                barrier_t = barrier_full
            else:
                barrier_t = barrier_full.subs(t_sym, float(t_val))
            last_im = render_panel(ax, syms, barrier_t, gamma, lam, 2.0,
                                   dx, dy, init_mid, L_init, U_init,
                                   unsafe_boxes, xlim, ylim, var_names,
                                   title=f'${lbl}$ at $t = {t_val:g}$')

    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.78])
    fig.colorbar(last_im, cax=cbar_ax, label='$B(x, t)$')

    proxies = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SAFE_FILL, alpha=0.6,
                      label=rf'$\{{B \leq \gamma\}}$ ($\gamma = {gamma:.4g}$)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL_LIGHT, alpha=0.45,
                      label=rf'$\{{B \geq \lambda\}}$ ($\lambda = {lam:.4g}$)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=INIT_FILL, edgecolor=INIT_EDGE,
                      linewidth=1.4, label=r'initial set $X_0$'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL, edgecolor=UNSAFE_EDGE,
                      alpha=0.85, label=r'unsafe set $X_u$'),
        plt.Line2D([0], [0], color=STREAM_COLOR, lw=1.0,
                   label=r'$f(x, b=2)$ flow'),
    ]
    fig.legend(handles=proxies, loc='lower center', ncol=3, fontsize=9,
               framealpha=0.95, bbox_to_anchor=(0.45, -0.005))
    fig.suptitle(f'CVDP23 finite-time barrier $B(x, t)$ at three '
                 f'$t \\in [0, {T_horizon:g}]$ slices  '
                 f'($\\gamma = {gamma:.4g}$, $\\lambda - \\gamma = '
                 f'{lam-gamma:+.2e}$)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 0.91, 0.94])

    out_path = os.path.join(RESULTS, 'CVDP23_finite_time_topology.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  ok  {out_path}  (gamma={gamma}, lambda={lam}, T={T_horizon})')


if __name__ == '__main__':
    main()
