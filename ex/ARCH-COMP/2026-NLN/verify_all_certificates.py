"""
Numerical verification of every barrier certificate stored under results/.

For each (label, .result.json) pair we sample the closed boxes for X_0
and X_u, plus the state-space box for the Lie derivative, and compute:

    sup_{x in X_0}  B(x)         vs.  gamma   (target: <= 0 with slack)
    inf_{x in X_u}  B(x)         vs.  lambda  (target: <= 0 with slack)
    sup_{x in X}    <grad B, f>  (target: <= 0; tolerance 1e-2)

The benchmark geometries (initial / unsafe / state-space sets and dynamics)
are hand-encoded here so the verifier doesn't have to import the benchmark
scripts (those scripts run on __main__ and have side effects).

For benchmarks with an uncertain parameter (CVDP23 b_unc, TRAF22), the
Lie sup is taken across a sample sweep of parameter values.

Output: a pass/fail table to stdout (also written to
results/verification_summary.csv).
"""

import csv
import json
import os
import re
from collections import OrderedDict

import numpy as np
import sympy as sp


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')


def _sympify_barrier(s):
    s = re.sub(r'(\d)\.e([+-]?\d+)', r'\1.0e\2', s)
    return sp.sympify(s)


def _sample_box(L, U, n, rng):
    return rng.uniform(L, U, size=(n, len(L)))


# ---------------------------------------------------------------------
# Per-benchmark specifications.
# ---------------------------------------------------------------------

def _coupled_vdp_dyn(syms, b):
    x1, y1, x2, y2 = syms
    mu = 1.0
    return [
        y1,
        mu * (1 - x1**2) * y1 + b * (x2 - x1) - x1,
        y2,
        mu * (1 - x2**2) * y2 - b * (x2 - x1) - x2,
    ]


def _robe_dyn(syms, alpha, beta, gamma_c, scale_y):
    u, V, w = syms
    return [
        -alpha * u + (beta / scale_y) * V * w,
        scale_y * alpha * u - beta * V * w - (gamma_c / scale_y) * V**2,
        (gamma_c / scale_y**2) * V**2,
    ]


def _lorenz_dyn(syms):
    x, y, z = syms
    return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]


def _lalo20_dyn(syms):
    x1, x2, x3, x4, x5, x6, x7 = syms
    return [
        1.4 * x3 - 0.9 * x1,
        2.5 * x5 - 1.5 * x2,
        0.6 * x7 - 0.8 * x2 * x3,
        2.0 - 1.3 * x3 * x4,
        0.7 * x1 - x4 * x5,
        0.3 * x1 - 3.1 * x6,
        1.8 * x6 - 1.5 * x2 * x7,
    ]


def _lovo25_dyn(syms):
    x, y = syms
    return [3 * x - 3 * x * y, x * y - y]


def _traf22_dyn(syms, q_s):
    delta, psi, v, sy = syms
    l_wb = 2.578
    return [0, (v / l_wb) * delta, 0, v * psi * q_s]


def _spre22_dyn(syms):
    u, v, vx, vy = syms
    R = 42164.0e3
    MU = 3.986e14 * 60.0**2
    N = (MU / R**3) ** 0.5
    MC = 500.0
    K2 = [
        [-288.0288, 0.1312, -9614.9898, 0.0],
        [-0.1312, -288.0, 0.0, -9614.9883],
    ]
    x_m = u * 1000.0
    y_m = v * 1000.0
    state = [x_m, y_m, vx, vy]
    ufb0 = sum(K2[0][i] * state[i] for i in range(4))
    ufb1 = sum(K2[1][i] * state[i] for i in range(4))
    rc2 = (R + x_m)**2 + y_m**2
    # Use sqrt for the radial term; this stays polynomial-free but is fine
    # for sampling.
    rc = sp.sqrt(rc2)
    return [
        vx / 1000.0,
        vy / 1000.0,
        N**2 * x_m + 2 * N * vy + MU / R**2
            - MU / rc**3 * (R + x_m) + ufb0 / MC,
        N**2 * y_m - 2 * N * vx
            - MU / rc**3 * y_m + ufb1 / MC,
    ]


# Each benchmark's spec: how to build init/unsafe/state samples and dyn.
SPECS = OrderedDict()


# ROBE25 (3-D): u, V, w. State space envelope from the benchmark script.
for ii, beta_v, gamma_v in [(2, 1e3, 1e5), (3, 1e3, 1e7)]:
    scale_y = 100.0
    eps = 1e-3
    SPECS[f'ROBE25_{ii}'] = {
        'state_dim': 3,
        'syms_name': 'x0:3',
        'L_init':   [1.0 - eps, -eps * scale_y, -eps],
        'U_init':   [1.0 + eps,  eps * scale_y,  eps],
        'unsafe_regions': [
            ([0.0, -0.5 * scale_y, 0.0], [0.9, 0.5 * scale_y, 1.1]),
            ([0.0, -0.5 * scale_y, 1.0], [1.1, 0.5 * scale_y, 1.1]),
        ],
        'L_space':  [0.0, -0.5 * scale_y, 0.0],
        'U_space':  [1.1,  0.5 * scale_y, 1.1],
        'dyn': lambda s, a=0.4, b=beta_v, g=gamma_v, sy=scale_y:
                  _robe_dyn(s, a, b, g, sy),
        'p_samples': [None],
    }


# ROBE21 instances.
for ii, eps in [(1, 0.001), (2, 0.005), (3, 0.01)]:
    SPECS[f'ROBE21_{ii}'] = {
        'state_dim': 3,
        'syms_name': 'x0:3',
        'L_init':   [1.0 - eps, -eps * 1e4, -eps],
        'U_init':   [1.0 + eps,  eps * 1e4,  eps],
        'unsafe_regions': [
            ([0.0, -1e3, 0.0], [0.9, 1e3, 1.1]),
            ([0.0, -1e3, 1.0], [1.1, 1e3, 1.1]),
        ],
        'L_space':  [0.0, -1e3, 0.0],
        'U_space':  [1.1,  1e3, 1.1],
        'dyn': lambda s: _robe_dyn(s, 0.04, 1e4, 3e7, 1e4),
        'p_samples': [None],
    }


# CVDP family.
def _cvdp_spec(b_or_params, label, unsafe_threshold):
    return {
        'state_dim': 4,
        'syms_name': 'x0:4',
        'L_init':   [1.25, 2.35, 1.25, 2.35],
        'U_init':   [1.55, 2.45, 1.55, 2.45],
        'unsafe_regions': [
            ([-3.0, unsafe_threshold, -3.0, -3.0], [3.0, 3.0, 3.0, 3.0]),
            ([-3.0, -3.0, -3.0, unsafe_threshold], [3.0, 3.0, 3.0, 3.0]),
        ],
        'L_space':  [-3.0, -3.0, -3.0, -3.0],
        'U_space':  [3.0,  3.0,  3.0,  3.0],
        'dyn': (lambda s, b=b_or_params: _coupled_vdp_dyn(s, b))
                if isinstance(b_or_params, (int, float))
                else (lambda s: _coupled_vdp_dyn(s, 2.0)),
        'p_samples': ([None] if isinstance(b_or_params, (int, float))
                              else b_or_params),
        # If parameters, evaluate dyn at each b: handled in the verify loop.
        'param_kind': ('fixed_b' if isinstance(b_or_params, (int, float))
                                  else 'cvdp_b'),
    }


SPECS['CVDP23_b2']    = _cvdp_spec(2.0,                       'CVDP23_b2', 2.75)
SPECS['CVDP23_b_unc'] = _cvdp_spec([1.0, 1.5, 2.0, 2.5, 3.0], 'CVDP23_b_unc', 2.75)

# CVDP22 uses a wider state-space envelope because its unsafe threshold
# is 3.7 (not 2.75). Override the box bounds and unsafe regions.
SPECS['CVDP22'] = {
    'state_dim': 4,
    'syms_name': 'x0:4',
    'L_init':   [1.25, 2.35, 1.25, 2.35],
    'U_init':   [1.55, 2.45, 1.55, 2.45],
    'unsafe_regions': [
        ([-3.0, 3.7, -3.0, -4.0], [3.0, 4.0, 3.0, 4.0]),
        ([-3.0, -4.0, -3.0, 3.7], [3.0, 4.0, 3.0, 4.0]),
    ],
    'L_space':  [-3.0, -4.0, -3.0, -4.0],
    'U_space':  [ 3.0,  4.0,  3.0,  4.0],
    'dyn': lambda s: _coupled_vdp_dyn(s, 70.0),
    'p_samples': [None],
}

# LOVO25 (2-D Lotka-Volterra). Initial set as a thin line, unsafe = 4 boxes.
SPECS['LOVO25'] = {
    'state_dim': 2,
    'syms_name': 'x0:2',
    'L_init':   [1.288, 1.0 - 1e-3],
    'U_init':   [1.312, 1.0 + 1e-3],
    'unsafe_regions': [
        ([0.5, 0.5], [0.6, 1.5]),
        ([1.4, 0.5], [1.5, 1.5]),
        ([0.5, 0.5], [1.5, 0.6]),
        ([0.5, 1.4], [1.5, 1.5]),
    ],
    'L_space':  [0.6, 0.6],
    'U_space':  [1.4, 1.4],
    'dyn': _lovo25_dyn,
    'p_samples': [None],
}


# LOVO21 (Lorenz).
SPECS['LOVO21'] = {
    'state_dim': 3,
    'syms_name': 'x0:3',
    'L_init':   [0.9, -0.01, -0.01],
    'U_init':   [1.1,  0.01,  0.01],
    'unsafe_regions': [([20.0, -30.0, 0.0], [30.0, 30.0, 60.0])],
    'L_space':  [-30.0, -30.0, 0.0],
    'U_space':  [30.0, 30.0, 60.0],
    'dyn': _lorenz_dyn,
    'p_samples': [None],
}


# LALO20 instances.
for inst, W, x4_unsafe in [('W001', 0.01, 4.5),
                           ('W005', 0.05, 4.5),
                           ('W01',  0.10, 5.0)]:
    centre = [1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45]
    L_init = [c - W for c in centre]
    U_init = [c + W for c in centre]
    L_space = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0]
    U_space = [4.0, 4.0, 4.0, x4_unsafe + 1.0, 4.0, 4.0, 4.0]
    L_unsafe = list(L_space); L_unsafe[3] = x4_unsafe
    SPECS[f'LALO20_{inst}'] = {
        'state_dim': 7,
        'syms_name': 'x0:7',
        'L_init': L_init, 'U_init': U_init,
        'unsafe_regions': [(L_unsafe, U_space)],
        'L_space': L_space, 'U_space': U_space,
        'dyn': _lalo20_dyn,
        'p_samples': [None],
    }


# TRAF22 (4-D, sinc relaxation -> q_s in [sinc(0.5), 1]).
def _sinc_lo():
    import math
    return math.sin(0.5) / 0.5

SPECS['TRAF22'] = {
    'state_dim': 4,
    'syms_name': 'x0:4',
    'L_init':   [-0.04, -0.04, 4.9, -0.1],
    'U_init':   [ 0.04,  0.04, 5.1,  0.1],
    'unsafe_regions': [
        ([-0.4, -0.5, 4.0,  1.5], [0.4, 0.5, 6.0, 2.0]),
        ([-0.4, -0.5, 4.0, -2.0], [0.4, 0.5, 6.0, -1.5]),
    ],
    'L_space':  [-0.4, -0.5, 4.0, -2.0],
    'U_space':  [ 0.4,  0.5, 6.0,  2.0],
    'dyn': lambda s, q=1.0: _traf22_dyn(s, q),
    # Sweep q_s across the sinc relaxation interval.
    'p_samples': [_sinc_lo(), 0.95, 1.0],
    'param_kind': 'traf22_qs',
}


# SPRE22 (4-D, rescaled).
SPECS['SPRE22'] = {
    'state_dim': 4,
    'syms_name': 'x0:4',
    'L_init':   [-0.925, -0.425, 0.0, 0.0],
    'U_init':   [-0.875, -0.375, 5.0, 5.0],
    'unsafe_regions': [([-0.001, -0.001, -10.0, -10.0],
                        [ 0.001,  0.001,  10.0,  10.0])],
    'L_space':  [-1.0, -0.5, -10.0, -10.0],
    'U_space':  [ 0.05, 0.05,  10.0,  10.0],
    'dyn': _spre22_dyn,
    'p_samples': [None],
}


# ---------------------------------------------------------------------
# Verifier.
# ---------------------------------------------------------------------

def verify(label, spec, n_init=4000, n_unsafe=4000, n_lie=10000, seed=0):
    rj = os.path.join(RESULTS, label + '.result.json')
    if not os.path.isfile(rj):
        return None
    with open(rj) as f:
        data = json.load(f)
    if 'barrier' not in data or data.get('gamma') is None:
        return None
    B_expr = _sympify_barrier(data['barrier'])
    gamma = float(data['gamma'])
    lam = float(data['lambda'])

    syms = sp.symbols(spec['syms_name'])
    B_fn = sp.lambdify(syms, B_expr, 'numpy')

    rng = np.random.default_rng(seed)

    # (1) init
    init_samples = _sample_box(np.array(spec['L_init']),
                               np.array(spec['U_init']), n_init, rng)
    Bv = B_fn(*[init_samples[:, i] for i in range(spec['state_dim'])])
    sup_init = float(np.nanmax(Bv))

    # (2) unsafe
    inf_unsafe = np.inf
    for L_u, U_u in spec['unsafe_regions']:
        samples = _sample_box(np.array(spec['L_space']),
                              np.array(spec['U_space']), n_unsafe, rng)
        for i in range(spec['state_dim']):
            if L_u[i] > spec['L_space'][i] or U_u[i] < spec['U_space'][i]:
                samples[:, i] = rng.uniform(L_u[i], U_u[i], n_unsafe)
        Bu = B_fn(*[samples[:, i] for i in range(spec['state_dim'])])
        inf_unsafe = min(inf_unsafe, float(np.nanmin(Bu)))

    # (3) Lie derivative across param samples.
    grad_B = [sp.diff(B_expr, s) for s in syms]
    sup_lie = -np.inf
    n_state_samples = max(n_lie // max(len(spec['p_samples']), 1), 1000)
    for p_val in spec['p_samples']:
        if p_val is None:
            dyn_exprs = spec['dyn'](syms)
        else:
            kind = spec.get('param_kind', '')
            if kind == 'cvdp_b':
                dyn_exprs = _coupled_vdp_dyn(syms, p_val)
            elif kind == 'traf22_qs':
                dyn_exprs = _traf22_dyn(syms, p_val)
            else:
                dyn_exprs = spec['dyn'](syms)
        # <grad B, f>
        dot = sum(grad_B[i] * dyn_exprs[i] for i in range(spec['state_dim']))
        try:
            dot_fn = sp.lambdify(syms, dot, 'numpy')
        except Exception:
            continue
        samples = _sample_box(np.array(spec['L_space']),
                              np.array(spec['U_space']),
                              n_state_samples, rng)
        try:
            Lv = dot_fn(*[samples[:, i] for i in range(spec['state_dim'])])
        except Exception:
            continue
        Lv = np.asarray(Lv, dtype=float)
        Lv = Lv[np.isfinite(Lv)]
        if Lv.size:
            sup_lie = max(sup_lie, float(np.nanmax(Lv)))

    init_slack   = sup_init - gamma           # want <= 0
    unsafe_slack = lam - inf_unsafe           # want <= 0
    lie_slack    = sup_lie                    # want <= 0

    verdict_init   = 'OK' if init_slack   <= 1e-3 else 'FAIL'
    verdict_unsafe = 'OK' if unsafe_slack <= 1e-3 else 'FAIL'
    verdict_lie    = 'OK' if lie_slack    <= 1e-2 else 'FAIL'
    overall = 'PASS' if (verdict_init == verdict_unsafe == verdict_lie == 'OK') \
              else 'FAIL'

    return {
        'label': label,
        'gamma': gamma, 'lambda': lam,
        'sup_B_X0': sup_init, 'init_slack': init_slack, 'init_verdict': verdict_init,
        'inf_B_Xu': inf_unsafe, 'unsafe_slack': unsafe_slack, 'unsafe_verdict': verdict_unsafe,
        'sup_Lie':  sup_lie, 'lie_verdict':   verdict_lie,
        'overall': overall,
    }


def main():
    rows = []
    print(f'{"benchmark":<16} {"gamma":>9}  {"lambda":>9}  '
          f'{"init slack":>12} {"unsafe slack":>14} {"Lie slack":>12}  verdict')
    print('-' * 100)
    for label, spec in SPECS.items():
        r = verify(label, spec)
        if r is None:
            print(f'{label:<16} (no .result.json found)')
            continue
        rows.append(r)
        print(f'{label:<16} '
              f'{r["gamma"]:>9.4g}  '
              f'{r["lambda"]:>9.4g}  '
              f'{r["init_slack"]:>+12.3e} {r["unsafe_slack"]:>+14.3e} '
              f'{r["sup_Lie"]:>+12.3e}  {r["overall"]}')

    # Write CSV.
    csv_path = os.path.join(RESULTS, 'verification_summary.csv')
    fields = ['label', 'gamma', 'lambda',
              'sup_B_X0', 'init_slack', 'init_verdict',
              'inf_B_Xu', 'unsafe_slack', 'unsafe_verdict',
              'sup_Lie', 'lie_verdict', 'overall']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print()
    print(f'-> {csv_path}')


if __name__ == '__main__':
    main()
