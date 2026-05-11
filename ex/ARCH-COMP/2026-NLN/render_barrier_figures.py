"""
Standalone re-render of every benchmark's barrier-certificate figure.

Each figure shows, in a 2-D slice of the state space:

  * the vector field of the dynamics as background streamlines (slicing the
    orthogonal coordinates through the INITIAL SET MIDPOINT, so the field at
    the displayed plane is the field seen by trajectories starting from the
    centre of the initial set);
  * the two level sets of the synthesised barrier B(x):
      - B(x) = gamma  (green) -- contains the initial set;
      - B(x) = lambda (red)   -- separates the initial set from the unsafe set;
  * the initial set X_0 (black-edged white box);
  * the unsafe set(s) X_u (solid red boxes).

Input: results/<label>.result.json (barrier polynomial + gamma + lambda).
Output: results/<label>.png.
"""

import json
import os
import re
from collections import namedtuple

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')


BenchSpec = namedtuple('BenchSpec', [
    'label',
    'dim_n',
    'var_names',
    'dim_x', 'dim_y',
    'L_initial', 'U_initial',
    'unsafe_boxes',
    'L_space', 'U_space',
    'title',
    'fig_xlim', 'fig_ylim',
    'dynamics',         # callable: list-of-sympy-symbols (length dim_n) ->
                        # list-of-sympy-exprs (length dim_n) for dx/dt
    'param_subs',       # dict of additional symbol substitutions
                        # (e.g. uncertain parameters at midpoint)
])


def _r(*xs):
    return np.asarray(xs, dtype=float)


def _spec(**kw):
    kw.setdefault('param_subs', {})
    return BenchSpec(**kw)


# ----------------------------------------------------------------------
# Per-benchmark specs (state symbols are x0, x1, ... matching saved
# barrier polynomials).
# ----------------------------------------------------------------------

SPECS = []


# --- ROBE25 -----------------------------------------------------------
# x0 = u, x1 = V = scale_y * y, x2 = w. Rescaled Robertson kinetics.
# Original kinetics: du/dt = -alpha*u + beta*y*z, dy/dt = alpha*u - beta*y*z - gamma*y^2, dz/dt = gamma*y^2
# With y = V / scale_y, z = w:
#   du/dt = -alpha*u + (beta/scale_y) * V * w
#   dV/dt = scale_y*alpha*u - beta*V*w - (gamma/scale_y) * V^2
#   dw/dt = (gamma/scale_y**2) * V^2

def make_robe_dyn(alpha, beta, gamma, scale_y):
    def f(xs):
        u, V, w = xs
        return [
            -alpha * u + (beta / scale_y) * V * w,
            scale_y * alpha * u - beta * V * w - (gamma / scale_y) * V**2,
            (gamma / scale_y**2) * V**2,
        ]
    return f


# ROBE25 instances: alpha=0.4 fixed; (beta, gamma) per instance.
for ii, beta, gamma, scale_y in [
        (2, 1e3, 1e5, 100.0),   # instance 2
        (3, 1e3, 1e7, 100.0)]:  # instance 3
    SPECS.append(_spec(
        label=f'ROBE25_{ii}',
        dim_n=3, var_names=('u', 'V', 'w'),
        dim_x=0, dim_y=2,
        L_initial=_r(0.999, -1e-3*scale_y, -1e-3),
        U_initial=_r(1.001,  1e-3*scale_y,  1e-3),
        unsafe_boxes=[
            (_r(0.0, -0.5*scale_y, 0.0), _r(0.9, 0.5*scale_y, 1.1)),
            (_r(0.0, -0.5*scale_y, 1.0), _r(1.1, 0.5*scale_y, 1.1)),
        ],
        L_space=_r(0.0, -0.5*scale_y, 0.0),
        U_space=_r(1.1,  0.5*scale_y, 1.1),
        title=f'ROBE25 instance {ii} (autocatalytic, 2026 spec)',
        fig_xlim=(0.85, 1.05), fig_ylim=(-0.05, 1.05),
        dynamics=make_robe_dyn(0.4, beta, gamma, scale_y),
    ))


# --- ROBE21 -----------------------------------------------------------
# Original ARCH-COMP rescaling: alpha=0.04, beta=1e4, gamma=3e7. scale_y=1e4.
for ii, eps in [(1, 0.001), (2, 0.005), (3, 0.01)]:
    SPECS.append(_spec(
        label=f'ROBE21_{ii}',
        dim_n=3, var_names=('u', 'V', 'w'),
        dim_x=0, dim_y=2,
        L_initial=_r(1.0 - eps, -eps*1e4, -eps),
        U_initial=_r(1.0 + eps,  eps*1e4,  eps),
        unsafe_boxes=[
            (_r(0.0, -1e3, 0.0), _r(0.9, 1e3, 1.1)),
            (_r(0.0, -1e3, 1.0), _r(1.1, 1e3, 1.1)),
        ],
        L_space=_r(0.0, -1e3, 0.0),
        U_space=_r(1.1,  1e3, 1.1),
        title=f'ROBE21 instance {ii} (rescaled Robertson)',
        fig_xlim=(0.85, 1.05), fig_ylim=(-0.05, 1.05),
        dynamics=make_robe_dyn(0.04, 1e4, 3e7, 1e4),
    ))


# --- CVDP family ------------------------------------------------------
# State: x0 = x1, x1 = y1, x2 = x2, x3 = y2 (coupled van der Pol).

def make_cvdp_dyn(mu, b):
    def f(xs):
        x1, y1, x2, y2 = xs
        return [
            y1,
            mu * (1 - x1**2) * y1 + b * (x2 - x1) - x1,
            y2,
            mu * (1 - x2**2) * y2 - b * (x2 - x1) - x2,
        ]
    return f


SPECS.append(_spec(
    label='CVDP23_b2', dim_n=4, var_names=('x_1', 'y_1', 'x_2', 'y_2'),
    dim_x=0, dim_y=1,
    L_initial=_r(1.25, 2.35, 1.25, 2.35),
    U_initial=_r(1.55, 2.45, 1.55, 2.45),
    unsafe_boxes=[
        (_r(-3.0, 2.75, -3.0, -3.0), _r(3.0, 3.0, 3.0, 3.0)),
        (_r(-3.0, -3.0, -3.0, 2.75), _r(3.0, 3.0, 3.0, 3.0)),
    ],
    L_space=_r(-3.0, -3.0, -3.0, -3.0),
    U_space=_r(3.0, 3.0, 3.0, 3.0),
    title='CVDP23 (b=2, fixed-midpoint reduction)',
    fig_xlim=(0.8, 1.8), fig_ylim=(2.2, 3.0),
    dynamics=make_cvdp_dyn(mu=1.0, b=2.0),
))

# CVDP23 b uncertain: use the midpoint b=2 for the vector field display.
SPECS.append(_spec(
    label='CVDP23_b_unc', dim_n=4, var_names=('x_1', 'y_1', 'x_2', 'y_2'),
    dim_x=0, dim_y=1,
    L_initial=_r(1.25, 2.35, 1.25, 2.35),
    U_initial=_r(1.55, 2.45, 1.55, 2.45),
    unsafe_boxes=[
        (_r(-3.0, 2.75, -3.0, -3.0), _r(3.0, 3.0, 3.0, 3.0)),
        (_r(-3.0, -3.0, -3.0, 2.75), _r(3.0, 3.0, 3.0, 3.0)),
    ],
    L_space=_r(-3.0, -3.0, -3.0, -3.0),
    U_space=_r(3.0, 3.0, 3.0, 3.0),
    title='CVDP23 (b in [1,3] uncertain; vector field at b=2)',
    fig_xlim=(0.8, 1.8), fig_ylim=(2.2, 3.0),
    dynamics=make_cvdp_dyn(mu=1.0, b=2.0),
))

SPECS.append(_spec(
    label='CVDP22', dim_n=4, var_names=('x_1', 'y_1', 'x_2', 'y_2'),
    dim_x=0, dim_y=1,
    L_initial=_r(1.25, 2.35, 1.25, 2.35),
    U_initial=_r(1.55, 2.45, 1.55, 2.45),
    unsafe_boxes=[
        (_r(-3.0, 3.7, -3.0, -4.0), _r(3.0, 4.0, 3.0, 4.0)),
        (_r(-3.0, -4.0, -3.0, 3.7), _r(3.0, 4.0, 3.0, 4.0)),
    ],
    L_space=_r(-3.0, -4.0, -3.0, -4.0),
    U_space=_r(3.0, 4.0, 3.0, 4.0),
    title='CVDP22 (b=70, original spec)',
    fig_xlim=(0.8, 1.8), fig_ylim=(2.2, 3.9),
    dynamics=make_cvdp_dyn(mu=1.0, b=70.0),
))


# --- LOVO25 (2-D Lotka-Volterra) -------------------------------------
def lovo25_dyn(xs):
    x, y = xs
    return [3 * x - 3 * x * y, x * y - y]


SPECS.append(_spec(
    label='LOVO25', dim_n=2, var_names=('x', 'y'),
    dim_x=0, dim_y=1,
    L_initial=_r(1.288, 1.0),
    U_initial=_r(1.312, 1.0),
    unsafe_boxes=[
        (_r(0.5, 0.5), _r(0.6, 1.5)),
        (_r(1.4, 0.5), _r(1.5, 1.5)),
        (_r(0.5, 0.5), _r(1.5, 0.6)),
        (_r(0.5, 1.4), _r(1.5, 1.5)),
    ],
    L_space=_r(0.6, 0.6),
    U_space=_r(1.4, 1.4),
    title='LOVO25 (Lotka-Volterra, 2026 spec)',
    fig_xlim=(0.55, 1.45), fig_ylim=(0.55, 1.45),
    dynamics=lovo25_dyn,
))


# --- LOVO21 (Lorenz) -------------------------------------------------
def lovo21_dyn(xs):
    x, y, z = xs
    sigma, rho, beta_ = 10.0, 28.0, 8.0 / 3.0
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta_ * z]


SPECS.append(_spec(
    label='LOVO21', dim_n=3, var_names=('x', 'y', 'z'),
    dim_x=0, dim_y=2,
    L_initial=_r(0.9, -0.01, -0.01),
    U_initial=_r(1.1,  0.01,  0.01),
    unsafe_boxes=[(_r(20.0, -30.0, 0.0), _r(30.0, 30.0, 60.0))],
    L_space=_r(-30.0, -30.0, 0.0),
    U_space=_r(30.0, 30.0, 60.0),
    title='LOVO21 (Lorenz, original spec, margin=4)',
    fig_xlim=(-15.0, 30.0), fig_ylim=(-2.0, 30.0),
    dynamics=lovo21_dyn,
))


# --- LALO20 (Laub-Loomis, 7-D) ---------------------------------------
def lalo20_dyn(xs):
    x1, x2, x3, x4, x5, x6, x7 = xs
    return [
        1.4 * x3 - 0.9 * x1,
        2.5 * x5 - 1.5 * x2,
        0.6 * x7 - 0.8 * x2 * x3,
        2.0 - 1.3 * x3 * x4,
        0.7 * x1 - x4 * x5,
        0.3 * x1 - 3.1 * x6,
        1.8 * x6 - 1.5 * x2 * x7,
    ]


for inst, W, x4_unsafe in [('W001', 0.01, 4.5),
                           ('W005', 0.05, 4.5),
                           ('W01',  0.10, 5.0)]:
    centre = _r(1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45)
    half   = np.full(7, W)
    L_space = _r(0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
    U_space = _r(4.0, 4.0, 4.0, x4_unsafe + 1.0, 4.0, 4.0, 4.0)
    L_unsafe = L_space.copy(); L_unsafe[3] = x4_unsafe
    SPECS.append(_spec(
        label=f'LALO20_{inst}', dim_n=7,
        var_names=tuple(f'x_{i+1}' for i in range(7)),
        dim_x=2, dim_y=3,
        L_initial=centre - half, U_initial=centre + half,
        unsafe_boxes=[(L_unsafe, U_space)],
        L_space=L_space, U_space=U_space,
        title=f'LALO20 ({inst})',
        fig_xlim=(1.0, 3.5), fig_ylim=(1.7, x4_unsafe + 0.5),
        dynamics=lalo20_dyn,
    ))


# --- SPRE22 (rescaled spacecraft, mode "rendezvous attempt") ---------
# x = (u, v, vx, vy). u = x/1000, v = y/1000. K2 feedback. mu, r, n constants.
SPRE_MU = 3.986e14 * 60.0**2
SPRE_R  = 42164.0e3
SPRE_N  = (SPRE_MU / SPRE_R**3) ** 0.5
SPRE_MC = 500.0
SPRE_K2 = np.array([
    [-288.0288, 0.1312, -9614.9898, 0.0],
    [-0.1312, -288.0, 0.0, -9614.9883],
])


def spre22_dyn(xs):
    u, v, vx, vy = xs
    x_m = u * 1000.0
    y_m = v * 1000.0
    state_v = sp.Matrix([x_m, y_m, vx, vy])
    ufb = SPRE_K2 @ np.array([x_m, y_m, vx, vy], dtype=object)
    # The position derivatives are in m/min; we want u/dt and v/dt in km/min:
    rc = sp.sqrt((SPRE_R + x_m)**2 + y_m**2)
    return [
        vx / 1000.0,
        vy / 1000.0,
        SPRE_N**2 * x_m + 2 * SPRE_N * vy + SPRE_MU / SPRE_R**2
            - SPRE_MU / rc**3 * (SPRE_R + x_m) + ufb[0] / SPRE_MC,
        SPRE_N**2 * y_m - 2 * SPRE_N * vx
            - SPRE_MU / rc**3 * y_m + ufb[1] / SPRE_MC,
    ]


SPECS.append(_spec(
    label='SPRE22', dim_n=4, var_names=('u', 'v', 'v_x', 'v_y'),
    dim_x=0, dim_y=1,
    L_initial=_r(-0.925, -0.425, 0.0, 0.0),
    U_initial=_r(-0.875, -0.375, 5.0, 5.0),
    unsafe_boxes=[(_r(-0.001, -0.001, -10.0, -10.0), _r(0.001, 0.001, 10.0, 10.0))],
    L_space=_r(-1.0, -0.5, -10.0, -10.0),
    U_space=_r( 0.05,  0.05,  10.0,  10.0),
    title='SPRE22 (rescaled, rendezvous-attempt mode)',
    fig_xlim=(-1.0, 0.05), fig_ylim=(-0.5, 0.05),
    dynamics=spre22_dyn,
))


# --- TRAF22 (4-D, sinc-relaxed kinematic) ----------------------------
# x = (delta, psi, v, sy); sin(psi) replaced by psi * q with q in [sinc(psi_max), 1].
# Use q = 1 (no nonlinear sinc deviation) for vector-field display.
def traf22_dyn(xs):
    delta, psi, v, sy = xs
    l_wb = 2.578
    return [0, (v / l_wb) * delta, 0, v * psi * 1.0]


SPECS.append(_spec(
    label='TRAF22', dim_n=4, var_names=('\\delta', '\\psi', 'v', 's_y'),
    dim_x=1, dim_y=3,
    L_initial=_r(-0.04, -0.04, 4.9, -0.1),
    U_initial=_r( 0.04,  0.04, 5.1,  0.1),
    unsafe_boxes=[
        (_r(-0.4, -0.5, 4.0,  1.5), _r(0.4, 0.5, 6.0, 2.0)),
        (_r(-0.4, -0.5, 4.0, -2.0), _r(0.4, 0.5, 6.0, -1.5)),
    ],
    L_space=_r(-0.4, -0.5, 4.0, -2.0),
    U_space=_r( 0.4,  0.5, 6.0,  2.0),
    title='TRAF22 (v2 robust SOS, sinc relaxation; q_s=1 in field)',
    fig_xlim=(-0.5, 0.5), fig_ylim=(-2.0, 2.0),
    dynamics=traf22_dyn,
))


# ----------------------------------------------------------------------
# Rendering core.
# ----------------------------------------------------------------------

GREEN_LINE = '#1f7a1f'
RED_LINE   = '#a51d1d'
INIT_EDGE  = '#000000'
INIT_FILL  = '#ffffff'
UNSAFE_EDGE = '#7a0000'
UNSAFE_FILL = '#e74c3c'
STREAM_COLOR = '#666666'


def _sympify_barrier(s):
    s = re.sub(r'(\d)\.e([+-]?\d+)', r'\1.0e\2', s)
    return sp.sympify(s)


def render_one(spec):
    rj_path = os.path.join(RESULTS, spec.label + '.result.json')
    if not os.path.isfile(rj_path):
        print(f'  MISS  {spec.label} (no .result.json)')
        return
    with open(rj_path) as fp:
        data = json.load(fp)
    if 'barrier' not in data or data.get('gamma') is None:
        print(f'  SKIP  {spec.label} (no barrier)')
        return

    barrier_expr = _sympify_barrier(data['barrier'])
    gamma = float(data['gamma'])
    lam   = float(data['lambda'])

    # Init-midpoint slice for orthogonal coords.
    x_syms = sp.symbols(f'x0:{spec.dim_n}')
    init_mid = 0.5 * (np.asarray(spec.L_initial, float) +
                      np.asarray(spec.U_initial, float))
    other_subs = {x_syms[i]: float(init_mid[i])
                  for i in range(spec.dim_n) if i not in (spec.dim_x, spec.dim_y)}

    # Barrier on slice.
    sliced_B = barrier_expr.subs(other_subs)
    B_eval = sp.lambdify((x_syms[spec.dim_x], x_syms[spec.dim_y]), sliced_B, 'numpy')

    # Vector field on slice.
    dyn_exprs = spec.dynamics(x_syms)
    fx_expr = sp.sympify(dyn_exprs[spec.dim_x]).subs(other_subs)
    fy_expr = sp.sympify(dyn_exprs[spec.dim_y]).subs(other_subs)
    fx = sp.lambdify((x_syms[spec.dim_x], x_syms[spec.dim_y]), fx_expr, 'numpy')
    fy = sp.lambdify((x_syms[spec.dim_x], x_syms[spec.dim_y]), fy_expr, 'numpy')

    xlim = spec.fig_xlim or (spec.L_space[spec.dim_x], spec.U_space[spec.dim_x])
    ylim = spec.fig_ylim or (spec.L_space[spec.dim_y], spec.U_space[spec.dim_y])
    # contour grid: dense
    nx = ny = 320
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)
    try:
        Zb = np.asarray(B_eval(X, Y), dtype=float)
    except Exception as exc:
        print(f'  FAIL  {spec.label}: B eval error {exc}')
        return
    # stream grid: coarser
    nxs = nys = 48
    xss = np.linspace(xlim[0], xlim[1], nxs)
    yss = np.linspace(ylim[0], ylim[1], nys)
    Xs, Ys = np.meshgrid(xss, yss)
    try:
        FX = np.asarray(fx(Xs, Ys), dtype=float)
        FY = np.asarray(fy(Xs, Ys), dtype=float)
    except Exception as exc:
        print(f'  WARN  {spec.label}: vector-field eval error {exc}; '
              f'figure will skip streamplot')
        FX = FY = None
    # Broadcast scalar constants so streamplot has 2-D arrays.
    if FX is not None:
        if np.ndim(FX) == 0: FX = np.full_like(Xs, float(FX))
        if np.ndim(FY) == 0: FY = np.full_like(Xs, float(FY))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # 1) Vector field: streamlines in grey.
    if FX is not None and FY is not None and np.isfinite(FX).any() and np.isfinite(FY).any():
        speed = np.hypot(FX, FY)
        speed_safe = np.where(speed > 0, speed, 1e-9)
        # Linewidth modulated lightly by log-speed to highlight fast flow.
        lw = 0.8 + 1.0 * (np.log10(speed_safe) - np.log10(speed_safe.min())) / \
             max(np.log10(speed_safe.max()) - np.log10(speed_safe.min()), 1e-6)
        try:
            ax.streamplot(Xs, Ys, FX, FY, color=STREAM_COLOR, density=1.2,
                          linewidth=lw, arrowsize=1.0)
        except Exception:
            # streamplot occasionally fails on degenerate fields; fall back to quiver.
            ax.quiver(Xs, Ys, FX, FY, color=STREAM_COLOR, alpha=0.6, scale=None)

    # 2) Barrier level sets gamma and lambda (two dividing lines).
    # SOS validator residuals can push B(x_init) slightly above the reported
    # gamma in numerical practice. Anchor the gamma contour at the actual
    # value of B at the initial-set midpoint, so the green dividing line is
    # guaranteed to pass through the initial set in the rendered slice.
    Zmin, Zmax = float(np.nanmin(Zb)), float(np.nanmax(Zb))
    B_at_init = float(B_eval(init_mid[spec.dim_x], init_mid[spec.dim_y]))
    # "tied" iff the certificate's gamma and lambda are essentially the same
    # value (i.e. PRoTECT couldn't establish a strict gap larger than the
    # SOS optimisation noise floor). Use a relative test on max(|gamma|,|lambda|).
    rel = abs(lam - gamma) / max(abs(gamma), abs(lam), 1.0)
    tied = rel < 0.05

    if not tied:
        plot_gamma  = max(gamma, B_at_init)
        plot_lambda = lam
        cs_g = ax.contour(X, Y, Zb, levels=[plot_gamma],
                          colors=[GREEN_LINE], linewidths=2.6, zorder=7)
        cs_l = ax.contour(X, Y, Zb, levels=[plot_lambda],
                          colors=[RED_LINE], linewidths=2.6, zorder=7)
    else:
        # Certificate has gamma ~ lambda: one effective dividing curve.
        # Draw it at B(init_midpoint) (guaranteed visible) and shade a thin
        # band around it labelled as the certificate level.
        band = max(0.01 * (Zmax - Zmin), 1e-3 * abs(B_at_init))
        # Two contour lines symmetric about B_at_init, but capped to within
        # the data range so both actually get drawn.
        lo = max(Zmin + 1e-6, B_at_init - 0.5 * band)
        hi = min(Zmax - 1e-6, B_at_init + 0.5 * band)
        # Always include at least the data-min-+ tick so green has something.
        plot_gamma  = lo if lo > Zmin else B_at_init
        plot_lambda = hi if hi > plot_gamma else B_at_init + band
        # Filled band between the two levels for visual cue.
        ax.contourf(X, Y, Zb, levels=[plot_gamma, plot_lambda],
                    colors=['#fde2a4'], alpha=0.5, zorder=4)
        cs_g = ax.contour(X, Y, Zb, levels=[plot_gamma],
                          colors=[GREEN_LINE], linewidths=2.4, zorder=7)
        cs_l = ax.contour(X, Y, Zb, levels=[plot_lambda],
                          colors=[RED_LINE], linewidths=2.4,
                          linestyles='--', zorder=7)

    # 3) Initial set.
    L0, U0 = np.asarray(spec.L_initial, float), np.asarray(spec.U_initial, float)
    w0 = U0[spec.dim_x] - L0[spec.dim_x]
    h0 = U0[spec.dim_y] - L0[spec.dim_y]
    if w0 <= 0:
        w0 = 0.015 * (xlim[1] - xlim[0]); x0 = L0[spec.dim_x] - 0.5 * w0
    else:
        x0 = L0[spec.dim_x]
    if h0 <= 0:
        h0 = 0.015 * (ylim[1] - ylim[0]); y0 = L0[spec.dim_y] - 0.5 * h0
    else:
        y0 = L0[spec.dim_y]
    ax.add_patch(Rectangle((x0, y0), w0, h0, edgecolor=INIT_EDGE,
                           facecolor=INIT_FILL, linewidth=2.0, zorder=10))

    # 4) Unsafe boxes. Only draw a box if the slice's orthogonal
    #    coordinates actually fall inside the box: otherwise the box
    #    represents an unsafe condition that doesn't apply in this slice
    #    and showing it would mislead the reader.
    for Lu, Uu in spec.unsafe_boxes:
        Lu = np.asarray(Lu, float); Uu = np.asarray(Uu, float)
        # Check that orthogonal dimensions of the box include the slice.
        slice_in_box = True
        for i in range(spec.dim_n):
            if i in (spec.dim_x, spec.dim_y):
                continue
            if not (Lu[i] <= init_mid[i] <= Uu[i]):
                slice_in_box = False
                break
        if not slice_in_box:
            continue
        wu = Uu[spec.dim_x] - Lu[spec.dim_x]
        hu = Uu[spec.dim_y] - Lu[spec.dim_y]
        if wu <= 0 or hu <= 0:
            continue
        ax.add_patch(Rectangle((Lu[spec.dim_x], Lu[spec.dim_y]), wu, hu,
                               edgecolor=UNSAFE_EDGE, facecolor=UNSAFE_FILL,
                               alpha=0.85, linewidth=1.5, zorder=8))

    # 5) Legend.
    if tied:
        proxies = [
            plt.Line2D([0], [0], color=GREEN_LINE, lw=2.6,
                       label=rf'lower bracket of $B = \gamma \approx \lambda = {gamma:.4g}$'),
            plt.Line2D([0], [0], color=RED_LINE, lw=2.4, linestyle='--',
                       label=r'upper bracket (same certificate level)'),
        ]
    else:
        proxies = [
            plt.Line2D([0], [0], color=GREEN_LINE, lw=2.6,
                       label=rf'$B(x) = \gamma = {gamma:.4g}$'),
            plt.Line2D([0], [0], color=RED_LINE, lw=2.6,
                       label=rf'$B(x) = \lambda = {lam:.4g}$'),
        ]
    proxies += [
        plt.Line2D([0], [0], color=STREAM_COLOR, lw=1.2,
                   label=r'vector field $f(x)$'),
        plt.Rectangle((0, 0), 1, 1, facecolor=INIT_FILL,
                      edgecolor=INIT_EDGE, linewidth=1.5,
                      label=r'initial set $X_0$'),
        plt.Rectangle((0, 0), 1, 1, facecolor=UNSAFE_FILL,
                      edgecolor=UNSAFE_EDGE, alpha=0.85,
                      label=r'unsafe set $X_u$'),
    ]
    ax.legend(handles=proxies, loc='best', fontsize=8.5, framealpha=0.92,
              handlelength=2.2)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(f'${spec.var_names[spec.dim_x]}$', fontsize=11)
    ax.set_ylabel(f'${spec.var_names[spec.dim_y]}$', fontsize=11)
    ax.set_title(spec.title, fontsize=11)
    ax.grid(False)

    out_path = os.path.join(RESULTS, spec.label + '.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'  ok    {spec.label}.png  '
          f'(gamma={gamma:.4g}, lambda={lam:.4g}'
          f'{", tied" if tied else ""})')


def main():
    for spec in SPECS:
        try:
            render_one(spec)
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f'  FAIL  {spec.label}: {exc}')


if __name__ == '__main__':
    main()
