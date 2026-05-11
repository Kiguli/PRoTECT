"""
Topology-style figures for the benchmarks that pass the strict (1e-8)
pointwise validator either as 'clean' or 'warn':

  LALO20/W001, W005, W01       (clean, well-separated gamma/lambda)
  LOVO25                       (warn, near-tight)
  CVDP23_finite_time           (warn, paper spec b in [1,3], t in [0,7])
  CVDP23_finite_time_fixedB1   (warn, b = 1 fixed, t in [0,7])

Each figure shows, on its chosen 2-D slice(s):
  * heatmap of B(x) (or B(x, t) for finite-time)
  * filled SUBLEVEL SET {B <= gamma} (green) -- contains X_0
  * filled SUPERLEVEL SET {B >= lambda} (red light) -- contains X_u
  * initial set X_0 box (black-edged white)
  * unsafe set X_u boxes (red, drawn only when the slice intersects them)
  * vector field f(x[, p]) as faint streamlines
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


def _draw_panel(ax, syms, barrier, gamma, lam, dynamics_callable,
                dim_x, dim_y, init_mid, L_init, U_init,
                unsafe_boxes, xlim, ylim, var_names, title,
                streamline_density=0.9):
    """Render one topology panel and return the heatmap image for colorbar."""
    n = len(syms)
    other_subs = {syms[i]: float(init_mid[i])
                  for i in range(n) if i not in (dim_x, dim_y)}
    sliced_B = barrier.subs(other_subs)
    B_eval = sp.lambdify((syms[dim_x], syms[dim_y]), sliced_B, 'numpy')

    nx = ny = 280
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)
    Zb = np.asarray(B_eval(X, Y), dtype=float)

    # Vector field if dynamics provided.
    if dynamics_callable is not None:
        dyn = dynamics_callable(syms)
        fx_expr = sp.sympify(dyn[dim_x]).subs(other_subs)
        fy_expr = sp.sympify(dyn[dim_y]).subs(other_subs)
        fx = sp.lambdify((syms[dim_x], syms[dim_y]), fx_expr, 'numpy')
        fy = sp.lambdify((syms[dim_x], syms[dim_y]), fy_expr, 'numpy')
        nxs = 26
        Xs, Ys = np.meshgrid(np.linspace(xlim[0], xlim[1], nxs),
                            np.linspace(ylim[0], ylim[1], nxs))
        FX = np.asarray(fx(Xs, Ys), dtype=float)
        FY = np.asarray(fy(Xs, Ys), dtype=float)
        if FX.ndim == 0: FX = np.full_like(Xs, float(FX))
        if FY.ndim == 0: FY = np.full_like(Xs, float(FY))
    else:
        FX = FY = None

    im = ax.imshow(Zb, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   origin='lower', cmap='viridis', aspect='auto',
                   alpha=0.85, zorder=0)
    if FX is not None and np.isfinite(FX).any():
        try:
            ax.streamplot(Xs, Ys, FX, FY, color=STREAM_COLOR,
                          density=streamline_density,
                          linewidth=0.55, arrowsize=0.65, zorder=2)
        except Exception:
            ax.quiver(Xs, Ys, FX, FY, color=STREAM_COLOR, alpha=0.6, zorder=2)

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


def _shared_legend(fig, gamma, lam, extra_lines=()):
    proxies = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SAFE_FILL, alpha=0.6,
                      label=rf'$\{{B \leq \gamma\}}$ ($\gamma = {gamma:.4g}$)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL_LIGHT, alpha=0.45,
                      label=rf'$\{{B \geq \lambda\}}$ ($\lambda = {lam:.4g}$)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=INIT_FILL, edgecolor=INIT_EDGE,
                      linewidth=1.4, label=r'initial set $X_0$'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL, edgecolor=UNSAFE_EDGE,
                      alpha=0.85, label=r'unsafe set $X_u$'),
    ]
    for line in extra_lines:
        proxies.append(line)
    fig.legend(handles=proxies, loc='lower center',
               ncol=min(3, len(proxies)),
               fontsize=9, framealpha=0.95, bbox_to_anchor=(0.45, -0.005))


# ----------------------------------------------------------------------
# Benchmark renderers
# ----------------------------------------------------------------------

def _lalo20_dyn(s):
    x1, x2, x3, x4, x5, x6, x7 = s
    return [1.4 * x3 - 0.9 * x1,
            2.5 * x5 - 1.5 * x2,
            0.6 * x7 - 0.8 * x2 * x3,
            2.0 - 1.3 * x3 * x4,
            0.7 * x1 - x4 * x5,
            0.3 * x1 - 3.1 * x6,
            1.8 * x6 - 1.5 * x2 * x7]


def render_lalo20(inst):
    inst_map = {'W001': (0.01, 4.5),
                'W005': (0.05, 4.5),
                'W01':  (0.10, 5.0)}
    W, x4_unsafe = inst_map[inst]

    label = f'LALO20_{inst}'
    rj = os.path.join(RESULTS, label + '.result.json')
    if not os.path.isfile(rj):
        print(f'  MISS {label}')
        return
    d = json.load(open(rj))
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
    pairs = [(0, 3), (1, 3), (2, 3),
             (4, 3), (5, 3), (6, 3)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    last_im = None
    for ax, (dx, dy) in zip(axes.flat, pairs):
        xlim = (L_space[dx], U_space[dx])
        ylim = (L_space[dy], U_space[dy])
        last_im = _draw_panel(ax, syms, barrier, gamma, lam, _lalo20_dyn,
                              dx, dy, init_mid, L_init, U_init,
                              unsafe_boxes, xlim, ylim, var_names,
                              title=f'$({var_names[dx]},\\,{var_names[dy]})$ projection')

    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.78])
    fig.colorbar(last_im, cax=cbar_ax, label='$B(x)$')

    _shared_legend(fig, gamma, lam, extra_lines=[
        plt.Line2D([0], [0], color=STREAM_COLOR, lw=1.0, label=r'flow $f(x)$'),
    ])
    fig.suptitle(f'{label}: barrier $B(x_1, \\dots, x_7)$ across $(x_i, x_4)$ '
                 f'projections  '
                 f'($\\gamma = {gamma:.4g}$, $\\lambda = {lam:.4g}$, '
                 f'$X_u = \\{{x_4 \\geq {x4_unsafe}\\}}$)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 0.91, 0.94])

    out = os.path.join(RESULTS, label + '_topology.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  ok  {out}')


def _lovo25_dyn(s):
    x, y = s
    return [3 * x - 3 * x * y, x * y - y]


def render_lovo25():
    rj = os.path.join(RESULTS, 'LOVO25.result.json')
    if not os.path.isfile(rj):
        return
    d = json.load(open(rj))
    barrier = _sympify(d['barrier'])
    gamma = float(d['gamma']); lam = float(d['lambda'])

    syms = sp.symbols('x0:2')
    L_init = np.array([1.288, 1.0 - 1e-3]); U_init = np.array([1.312, 1.0 + 1e-3])
    init_mid = 0.5 * (L_init + U_init)
    unsafe_boxes = [
        ([0.5, 0.5], [0.6, 1.5]),
        ([1.4, 0.5], [1.5, 1.5]),
        ([0.5, 0.5], [1.5, 0.6]),
        ([0.5, 1.4], [1.5, 1.5]),
    ]
    var_names = ['x', 'y']
    xlim = ylim = (0.55, 1.45)

    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    im = _draw_panel(ax, syms, barrier, gamma, lam, _lovo25_dyn,
                     0, 1, init_mid, L_init, U_init,
                     unsafe_boxes, xlim, ylim, var_names,
                     title='LOVO25 barrier $B(x, y)$',
                     streamline_density=1.2)
    cbar = fig.colorbar(im, ax=ax, label='$B(x, y)$')
    _shared_legend(fig, gamma, lam, extra_lines=[
        plt.Line2D([0], [0], color=STREAM_COLOR, lw=1.0, label=r'flow $f(x, y)$'),
    ])
    fig.suptitle(f'LOVO25 (Lotka-Volterra):  $\\gamma = {gamma:.4g}$, '
                 f'$\\lambda = {lam:.4g}$, $\\lambda - \\gamma = '
                 f'{lam-gamma:+.2e}$',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out = os.path.join(RESULTS, 'LOVO25_topology.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  ok  {out}')


def _coupled_vdp_dyn(syms, b):
    x1, y1, x2, y2 = syms
    return [y1,
            (1 - x1**2) * y1 + b * (x2 - x1) - x1,
            y2,
            (1 - x2**2) * y2 - b * (x2 - x1) - x2]


def render_cvdp23_finite_time(label, b_eff, suptitle_extra=''):
    rj = os.path.join(RESULTS, label + '.result.json')
    if not os.path.isfile(rj):
        print(f'  MISS {label}')
        return
    d = json.load(open(rj))
    barrier_full = _sympify(d['barrier'])
    gamma = float(d['gamma']); lam = float(d['lambda'])
    T_horizon = float(d.get('T_horizon', 7.0))

    syms = sp.symbols('x0:4')
    free_extra = [s for s in barrier_full.free_symbols if s not in syms]
    if not free_extra:
        print(f'  WARN {label}: barrier has no time symbol')
        t_sym = None
    else:
        t_sym = free_extra[0]

    L_init = np.array([1.25, 2.35, 1.25, 2.35])
    U_init = np.array([1.55, 2.45, 1.55, 2.45])
    init_mid = 0.5 * (L_init + U_init)
    unsafe_boxes = [
        ([-3.0, 2.75, -3.0, -3.0], [3.0, 3.0, 3.0, 3.0]),
        ([-3.0, -3.0, -3.0, 2.75], [3.0, 3.0, 3.0, 3.0]),
    ]
    var_names = ['x_1', 'y_1', 'x_2', 'y_2']
    proj_rows = [(0, 1), (2, 3)]
    t_cols = [0.0, T_horizon / 2.0, T_horizon]
    xlim = ylim = (-3.0, 3.0)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8))
    last_im = None
    for row_i, (dx, dy) in enumerate(proj_rows):
        for col_i, t_val in enumerate(t_cols):
            ax = axes[row_i, col_i]
            barrier_t = barrier_full if t_sym is None \
                                       else barrier_full.subs(t_sym, float(t_val))
            last_im = _draw_panel(ax, syms, barrier_t, gamma, lam,
                                  lambda s, b=b_eff: _coupled_vdp_dyn(s, b),
                                  dx, dy, init_mid, L_init, U_init,
                                  unsafe_boxes, xlim, ylim, var_names,
                                  title=f'$({var_names[dx]},\\,{var_names[dy]})$ at $t = {t_val:g}$')

    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.78])
    fig.colorbar(last_im, cax=cbar_ax, label='$B(x, t)$')

    _shared_legend(fig, gamma, lam, extra_lines=[
        plt.Line2D([0], [0], color=STREAM_COLOR, lw=1.0,
                   label=rf'$f(x, b={b_eff:g})$ flow'),
    ])
    fig.suptitle(f'{label}: finite-time barrier $B(x, t)$ at three '
                 f'$t \\in [0, {T_horizon:g}]$ slices'
                 f' {suptitle_extra}  '
                 f'($\\gamma = {gamma:.4g}$, $\\lambda - \\gamma = '
                 f'{lam-gamma:+.2e}$)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 0.91, 0.94])
    out = os.path.join(RESULTS, label + '_topology.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  ok  {out}')


def main():
    print('LALO20:')
    for inst in ['W001', 'W005', 'W01']:
        render_lalo20(inst)
    print('LOVO25:')
    render_lovo25()
    print('CVDP23 finite-time:')
    render_cvdp23_finite_time('CVDP23_finite_time', b_eff=2.0,
                              suptitle_extra='(paper spec, $b \\in [1,3]$ uncertain)')
    render_cvdp23_finite_time('CVDP23_finite_time_fixedB1_d2_k2', b_eff=1.0,
                              suptitle_extra='(simplified, $b = 1$ fixed)')


if __name__ == '__main__':
    main()
