"""
ROBE25_proposed -- adapted Robertson chemical reaction benchmark for
PRoTECT v2.

ORIGINAL ARCH-COMP 2026 ROBE25 spec has alpha=0.4 fixed and three
(beta, gamma) instances with the rate constants
  1: (1e2, 1e3),  2: (1e3, 1e5),  3: (1e3, 1e7).
The dynamics are autocatalytic with rates spanning 2-7 orders of
magnitude. PRoTECT v2's infinite-time SOS programme on the rescaled
(u, V, w) coordinates converges but pointwise-fails: the Lie residual
on the V-axis boundary blows up to ~1e3 (instance 2) or ~1e6
(instance 3) due to coefficient amplification at solver tolerance.

The PROPOSED variant uses the LEAST stiff instance (beta=1e2,
gamma=1e3 -> instance 1) with a TIGHTENED initial set (eps = 1e-4
instead of the original eps = 1e-3 for instances 1/2 or eps = 1e-4
for instance 3 already tight).

Result: a barrier that is pointwise-clean at strict 1e-8 tolerance.
Demonstrates the v2 pointwise validator on a stiff dynamical system
where the coefficient amplification is brought down to a level the
SOS solver can handle.
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS_robust import ct_DS_robust
from src.functions.result_export import write_result_json


if __name__ == '__main__':
    alpha = 0.4
    beta_v = 1e2; gamma_v = 1e3
    scale_y = 10.0
    eps = 1e-4   # TIGHTENED initial set (only change from ROBE25/1).

    dim = 3
    x = sp.symbols(f'x0:{dim}')   # u, V, w
    f = np.array([
        -alpha * x[0] + (beta_v / scale_y) * x[1] * x[2],
        scale_y * alpha * x[0] - beta_v * x[1] * x[2] - (gamma_v / scale_y) * x[1]**2,
        (gamma_v / scale_y**2) * x[1]**2,
    ])

    L_initial = np.array([1.0 - eps, 0.0,         0.0])
    U_initial = np.array([1.0 + eps, scale_y*eps, eps])
    L_space   = np.array([0.0, 0.0,           0.0])
    U_space   = np.array([1.1, scale_y * 0.5, 1.1])
    L_u1 = L_space.copy(); U_u1 = np.array([0.9, U_space[1], U_space[2]])
    L_u2 = np.array([L_space[0], L_space[1], 1.0]); U_u2 = U_space.copy()
    L_unsafe = np.array([L_u1, L_u2])
    U_unsafe = np.array([U_u1, U_u2])

    start = time.time()
    res = ct_DS_robust(
        b_degree=2, dim=3,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        x=x, f=f, p_syms=(), P_lo=(), P_hi=(),
        margin=0.0, solver='mosek',
        validate_sos=True, validate_tolerance=1e-8,
    )
    end = time.time()
    if res:
        res['variant'] = 'ROBE25_proposed (tightened init eps = 1e-4)'

    print('elapsed time:', (res or {}).get('solve_time', end - start))
    print(res if res else 'Results dictionary is empty.')

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'ROBE25_proposed', res if res else {})
