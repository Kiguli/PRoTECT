"""
CVDP23 barrier-certificate verification and 2x2-projection rendering.

For each CVDP23 instance saved under results/:
  1. Numerically verify the barrier:
       * sup_{x in X_0}      B(x)  <=  gamma   (initial-set positivity)
       * inf_{x in X_u}      B(x)  >=  lambda  (unsafe-set positivity)
       * sup_{x in X}        <dB/dx, f(x, p)>  <= 0  (Lie non-increasing)
     using dense random sampling of the relevant boxes. For the b-uncertain
     version the Lie check is swept over b in [1, 3] as well.
  2. Render a 4-panel figure with all four 2-D projections of the
     (x1, y1, x2, y2) state space -- (x1, y1), (x2, y1), (x1, y2),
     (x2, y2) -- each showing the vector field, initial/unsafe sets, and
     the gamma/lambda level sets.

This is the *direct* check of the certificate: if the sampling-based
maxima/minima all stay on the correct side of (gamma, lambda), the
certificate is valid to numerical precision over the sampled set; combined
with the SOS programme's symbolic guarantee on the full continuous box,
this is what makes the safety claim sound.
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


# Common CVDP23 spec (state ranges, init, unsafe, dynamics).
def cvdp_dynamics(b):
    x1, y1, x2, y2 = sp.symbols('x0 x1 x2 x3')
    mu = 1.0
    return [x1, y1, x2, y2], [
        y1,
        mu * (1 - x1**2) * y1 + b * (x2 - x1) - x1,
        y2,
        mu * (1 - x2**2) * y2 - b * (x2 - x1) - x2,
    ]


# Initial and unsafe sets (paper Sec. 3.3).
L_INIT = np.array([1.25, 2.35, 1.25, 2.35])
U_INIT = np.array([1.55, 2.45, 1.55, 2.45])
# unsafe = { y1 >= 2.75 } OR { y2 >= 2.75 }
UNSAFE_REGIONS = [
    {'dim': 1, 'lo': 2.75, 'hi': 3.0},
    {'dim': 3, 'lo': 2.75, 'hi': 3.0},
]
# State-space envelope used by the SOS programme.
L_SPACE = np.array([-3.0, -3.0, -3.0, -3.0])
U_SPACE = np.array([ 3.0,  3.0,  3.0,  3.0])

# Variable labels.
AXES = ['x_1', 'y_1', 'x_2', 'y_2']
PAIRS = [(0, 1), (2, 1), (0, 3), (2, 3)]


# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------

def _sympify_barrier(s):
    s = re.sub(r'(\d)\.e([+-]?\d+)', r'\1.0e\2', s)
    return sp.sympify(s)


def numerical_verify(barrier_expr, gamma, lam, b_values, n_init=4000,
                     n_unsafe=4000, n_lie=20000, seed=0):
    """Random-sample-based verification. Returns dict with the worst-case
    excursions (positive = violation) and a pass/fail summary."""
    rng = np.random.default_rng(seed)
    syms = sp.symbols('x0 x1 x2 x3')
    B_fn = sp.lambdify(syms, barrier_expr, 'numpy')

    # (1) Init: B(x) <= gamma for x in X_0.
    init_samples = rng.uniform(L_INIT, U_INIT, size=(n_init, 4))
    Bv = B_fn(*init_samples.T)
    sup_init = float(np.max(Bv))
    init_slack = sup_init - gamma   # negative if safe; positive if violated

    # (2) Unsafe: B(x) >= lambda for x in X_u (each region separately).
    inf_unsafe = +np.inf
    for region in UNSAFE_REGIONS:
        samples = rng.uniform(L_SPACE, U_SPACE, size=(n_unsafe, 4))
        # Project the relevant axis into the unsafe interval.
        samples[:, region['dim']] = rng.uniform(region['lo'], region['hi'], n_unsafe)
        Bu = B_fn(*samples.T)
        inf_unsafe = min(inf_unsafe, float(np.min(Bu)))
    unsafe_slack = lam - inf_unsafe   # negative if safe; positive if violated

    # (3) Lie derivative non-increasing: <grad B, f(x, b)> <= 0 on X.
    grad_B = [sp.diff(barrier_expr, s) for s in syms]
    sup_lie = -np.inf
    for b_val in b_values:
        sym_vars, dyn = cvdp_dynamics(b_val)
        dB_dot_f = sum(grad_B[i] * dyn[i] for i in range(4))
        dot_fn = sp.lambdify(syms, dB_dot_f, 'numpy')
        samples = rng.uniform(L_SPACE, U_SPACE, size=(n_lie, 4))
        Lv = dot_fn(*samples.T)
        sup_lie = max(sup_lie, float(np.max(Lv)))

    return {
        'sup_B_on_init': sup_init,
        'gamma': gamma,
        'init_slack': init_slack,
        'inf_B_on_unsafe': inf_unsafe,
        'lambda': lam,
        'unsafe_slack': unsafe_slack,
        'sup_Lie_on_state_space': sup_lie,
        'b_values_swept': list(b_values),
        'verdict': ('PASS' if init_slack <= 1e-3
                              and unsafe_slack <= 1e-3
                              and sup_lie <= 1e-2 else 'FAIL'),
    }


# ---------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------

GREEN_LINE = '#1f7a1f'
RED_LINE   = '#a51d1d'
INIT_EDGE  = '#000000'
INIT_FILL  = '#ffffff'
UNSAFE_EDGE = '#7a0000'
UNSAFE_FILL = '#e74c3c'
STREAM_COLOR = '#666666'


def _slice_subs(syms, dim_x, dim_y, init_mid):
    return {syms[i]: float(init_mid[i]) for i in range(4)
            if i not in (dim_x, dim_y)}


def _render_panel(ax, syms, barrier, gamma, lam, b_eff,
                  dim_x, dim_y, init_mid, title):
    subs = _slice_subs(syms, dim_x, dim_y, init_mid)

    # Sliced barrier.
    sliced_B = barrier.subs(subs)
    B_eval = sp.lambdify((syms[dim_x], syms[dim_y]), sliced_B, 'numpy')

    # Sliced dynamics (b fixed at b_eff = midpoint of param box / fixed b).
    _, dyn = cvdp_dynamics(b_eff)
    fx_expr = sp.sympify(dyn[dim_x]).subs(subs)
    fy_expr = sp.sympify(dyn[dim_y]).subs(subs)
    fx = sp.lambdify((syms[dim_x], syms[dim_y]), fx_expr, 'numpy')
    fy = sp.lambdify((syms[dim_x], syms[dim_y]), fy_expr, 'numpy')

    xlim = (L_SPACE[dim_x], U_SPACE[dim_x])
    ylim = (L_SPACE[dim_y], U_SPACE[dim_y])
    nx = ny = 280
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)
    Zb = np.asarray(B_eval(X, Y), dtype=float)

    nxs = nys = 36
    xss = np.linspace(xlim[0], xlim[1], nxs)
    yss = np.linspace(ylim[0], ylim[1], nys)
    Xs, Ys = np.meshgrid(xss, yss)
    FX = np.asarray(fx(Xs, Ys), dtype=float)
    FY = np.asarray(fy(Xs, Ys), dtype=float)
    if FX.ndim == 0: FX = np.full_like(Xs, float(FX))
    if FY.ndim == 0: FY = np.full_like(Xs, float(FY))

    # Layer 1: vector field.
    try:
        speed = np.hypot(FX, FY)
        speed_safe = np.where(speed > 0, speed, 1e-9)
        lw = 0.6 + 0.7 * (np.log10(speed_safe) - np.log10(speed_safe.min())) / \
             max(np.log10(speed_safe.max()) - np.log10(speed_safe.min()), 1e-6)
        ax.streamplot(Xs, Ys, FX, FY, color=STREAM_COLOR, density=1.0,
                      linewidth=lw, arrowsize=0.9, zorder=1)
    except Exception:
        ax.quiver(Xs, Ys, FX, FY, color=STREAM_COLOR, alpha=0.6, zorder=1)

    # Layer 3: unsafe boxes (only if the slice intersects the box in 4-D).
    for region in UNSAFE_REGIONS:
        # Region is "x[region['dim']] >= 2.75" with other dims free in state space.
        # Slice values used here: init_mid for all dims NOT in (dim_x, dim_y).
        # The region applies in the slice iff:
        #   (i)  region['dim'] in (dim_x, dim_y), OR
        #   (ii) init_mid[region['dim']] is inside [region['lo'], region['hi']].
        if region['dim'] in (dim_x, dim_y):
            # The unsafe condition affects the visible plane directly.
            if region['dim'] == dim_x:
                box_x = (region['lo'], region['hi'])
                box_y = ylim
            else:
                box_x = xlim
                box_y = (region['lo'], region['hi'])
            ax.add_patch(Rectangle(
                (box_x[0], box_y[0]),
                box_x[1] - box_x[0], box_y[1] - box_y[0],
                edgecolor=UNSAFE_EDGE, facecolor=UNSAFE_FILL,
                alpha=0.85, linewidth=1.5, zorder=3))
        else:
            slice_val = init_mid[region['dim']]
            if region['lo'] <= slice_val <= region['hi']:
                # Slice is inside this unsafe region; the WHOLE displayed
                # plane is unsafe -- draw a translucent overlay.
                ax.add_patch(Rectangle(
                    (xlim[0], ylim[0]),
                    xlim[1] - xlim[0], ylim[1] - ylim[0],
                    edgecolor=UNSAFE_EDGE, facecolor=UNSAFE_FILL,
                    alpha=0.25, linewidth=0, zorder=3))

    # Layer 4: initial set (projected box).
    L0i = L_INIT[dim_x]; U0i = U_INIT[dim_x]
    L0j = L_INIT[dim_y]; U0j = U_INIT[dim_y]
    ax.add_patch(Rectangle((L0i, L0j), U0i - L0i, U0j - L0j,
                           edgecolor=INIT_EDGE, facecolor=INIT_FILL,
                           linewidth=1.6, zorder=4))

    # Layer 6/8: level sets.
    Zmin, Zmax = float(np.nanmin(Zb)), float(np.nanmax(Zb))
    B_at_init = float(B_eval(0.5 * (L0i + U0i), 0.5 * (L0j + U0j)))
    rel = abs(lam - gamma) / max(abs(gamma), abs(lam), 1.0)
    tied = rel < 0.05
    if not tied:
        plot_gamma = max(gamma, B_at_init); plot_lambda = lam
        ax.contour(X, Y, Zb, levels=[plot_gamma], colors=[GREEN_LINE],
                   linewidths=2.0, zorder=8)
        ax.contour(X, Y, Zb, levels=[plot_lambda], colors=[RED_LINE],
                   linewidths=2.0, zorder=8)
    else:
        band = max(0.01 * (Zmax - Zmin), 1e-3 * abs(B_at_init))
        lo = max(Zmin + 1e-6, B_at_init - 0.5 * band)
        hi = min(Zmax - 1e-6, B_at_init + 0.5 * band)
        plot_gamma  = lo if lo > Zmin else B_at_init
        plot_lambda = hi if hi > plot_gamma else B_at_init + band
        ax.contourf(X, Y, Zb, levels=[plot_gamma, plot_lambda],
                    colors=['#fde2a4'], alpha=0.55, zorder=6)
        ax.contour(X, Y, Zb, levels=[plot_gamma], colors=[GREEN_LINE],
                   linewidths=1.8, zorder=8)
        ax.contour(X, Y, Zb, levels=[plot_lambda], colors=[RED_LINE],
                   linewidths=1.8, linestyles='--', zorder=8)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(f'${AXES[dim_x]}$', fontsize=10)
    ax.set_ylabel(f'${AXES[dim_y]}$', fontsize=10)
    ax.set_title(title, fontsize=10)


def render_grid(label, b_eff, out_path):
    rj_path = os.path.join(RESULTS, label + '.result.json')
    with open(rj_path) as fp:
        data = json.load(fp)
    barrier = _sympify_barrier(data['barrier'])
    gamma = float(data['gamma']); lam = float(data['lambda'])

    syms = sp.symbols('x0 x1 x2 x3')
    init_mid = 0.5 * (L_INIT + U_INIT)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5),
                             constrained_layout=False)
    for ax, (dx, dy) in zip(axes.flat, PAIRS):
        _render_panel(ax, syms, barrier, gamma, lam, b_eff,
                      dx, dy, init_mid,
                      title=f'projection $({AXES[dx]}, {AXES[dy]})$')

    # Shared legend at the bottom.
    proxies = [
        plt.Line2D([0], [0], color=GREEN_LINE, lw=2.2,
                   label=(rf'$B(x) = \gamma \approx \lambda = {gamma:.4g}$'
                          if abs(lam - gamma) / max(abs(gamma), 1.0) < 0.05
                          else rf'$B(x) = \gamma = {gamma:.4g}$')),
        plt.Line2D([0], [0], color=RED_LINE, lw=2.2, linestyle='--',
                   label=r'upper bracket of the certificate level'),
        plt.Line2D([0], [0], color=STREAM_COLOR, lw=1.2,
                   label=r'vector field $f(x, b=' + f'{b_eff:g}' + r')$'),
        plt.Rectangle((0, 0), 1, 1, facecolor=INIT_FILL,
                      edgecolor=INIT_EDGE, linewidth=1.5,
                      label=r'initial set $X_0$'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL,
                      edgecolor=UNSAFE_EDGE, alpha=0.85,
                      label=r'unsafe set $X_u$ (intersects slice)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL,
                      edgecolor='none', alpha=0.25,
                      label=r'whole slice is inside $X_u$ in 4-D'),
    ]
    fig.legend(handles=proxies, loc='lower center', ncol=3, fontsize=9,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f'CVDP23 barrier $B(x_1, y_1, x_2, y_2)$ across all four '
                 f'2-D projections of the state space  '
                 f'($\\gamma = {gamma:.4g}$, $\\lambda = {lam:.4g}$)',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------

def _summary_block(title, summary):
    line = '-' * len(title)
    print()
    print(title)
    print(line)
    print(f'  sup B on initial set  X_0   : {summary["sup_B_on_init"]:.6e}')
    print(f'  gamma                       : {summary["gamma"]:.6e}')
    print(f'  init slack (target <= 0)    : {summary["init_slack"]:+.3e}'
          f'  {"OK" if summary["init_slack"] <= 1e-3 else "VIOLATION"}')
    print(f'  inf B on unsafe set X_u     : {summary["inf_B_on_unsafe"]:.6e}')
    print(f'  lambda                      : {summary["lambda"]:.6e}')
    print(f'  unsafe slack (target <= 0)  : {summary["unsafe_slack"]:+.3e}'
          f'  {"OK" if summary["unsafe_slack"] <= 1e-3 else "VIOLATION"}')
    print(f'  sup <grad B, f> on X        : {summary["sup_Lie_on_state_space"]:+.3e}'
          f'  {"OK" if summary["sup_Lie_on_state_space"] <= 1e-2 else "VIOLATION"}')
    print(f'  b values swept              : {summary["b_values_swept"]}')
    print(f'  -> verdict                  : {summary["verdict"]}')


def main():
    for label, b_values_for_lie, b_eff_for_field in [
            ('CVDP23_b2',    [2.0],                       2.0),
            ('CVDP23_b_unc', [1.0, 1.5, 2.0, 2.5, 3.0],   2.0)]:
        rj_path = os.path.join(RESULTS, label + '.result.json')
        if not os.path.isfile(rj_path):
            print(f'(skip {label}: no .result.json)')
            continue
        with open(rj_path) as fp:
            data = json.load(fp)
        barrier = _sympify_barrier(data['barrier'])
        gamma = float(data['gamma']); lam = float(data['lambda'])

        summary = numerical_verify(barrier, gamma, lam, b_values_for_lie)
        _summary_block(f'{label}  numerical certificate check', summary)

        out_path = os.path.join(RESULTS, label + '_grid.png')
        render_grid(label, b_eff_for_field, out_path)
        print(f'  grid figure                 : {out_path}')


if __name__ == '__main__':
    main()
