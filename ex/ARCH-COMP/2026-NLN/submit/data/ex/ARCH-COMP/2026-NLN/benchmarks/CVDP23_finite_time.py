"""
CVDP23 with the PAPER specification's bounded time horizon t in [0, 7].

Uses the v2 finite-time-horizon SOS solver (ct_DS_finite_time) which
searches for a time-augmented barrier B(x, t) = sum_k t^k B_k(x) and a
time-box S-procedure multiplier g_t(t) = t * (T - t). Combined with the
robust-parameter machinery, this addresses the FULL paper spec:

  x_{1,2}(0) in [1.25, 1.55],  y_{1,2}(0) in [2.35, 2.45]
  b in [1, 3]                   (uncertain parameter)
  unsafe: y_1 >= 2.75 OR y_2 >= 2.75
  horizon: t in [0, 7]

Source: ARCH-COMP 2026 Nonlinear Dynamics report, Sec. 3.3.
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS_finite_time import ct_DS_finite_time
from src.functions.solve_helpers import solve_finite_time_safety_problem
from src.functions.result_export import write_result_json


if __name__ == '__main__':
    dim = 4
    x = sp.symbols(f'x0:{dim}')             # x1, y1, x2, y2
    p = sp.symbols('b0:1')                  # uncertain b

    L_initial = np.array([1.25, 2.35, 1.25, 2.35])
    U_initial = np.array([1.55, 2.45, 1.55, 2.45])

    L_space = np.array([-3.0, -3.0, -3.0, -3.0])
    U_space = np.array([ 3.0,  3.0,  3.0,  3.0])

    # unsafe: y1 >= 2.75 OR y2 >= 2.75 (two boxes).
    L_u1 = L_space.copy(); L_u1[1] = 2.75
    U_u1 = U_space.copy()
    L_u2 = L_space.copy(); L_u2[3] = 2.75
    U_u2 = U_space.copy()
    L_unsafe = np.array([L_u1, L_u2])
    U_unsafe = np.array([U_u1, U_u2])

    P_lo = np.array([1.0])
    P_hi = np.array([3.0])
    b_param, = p

    mu = 1.0
    f = np.array([
        x[1],
        mu * (1 - x[0]**2) * x[1] + b_param * (x[2] - x[0]) - x[0],
        x[3],
        mu * (1 - x[2]**2) * x[3] - b_param * (x[2] - x[0]) - x[2],
    ])

    T_horizon = 7.0
    start = time.time()
    result, solver_used = solve_finite_time_safety_problem(
        degrees=[2, 4],            # B_k spatial degree sweep
        time_orders=[1, 2],        # time polynomial order sweep
        T_horizon=T_horizon,
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        p_syms=p, P_lo=P_lo, P_hi=P_hi,
        margin=0.0,
        validate_tolerance=1e-3,
    )
    end = time.time()
    if result:
        result['solver'] = solver_used
        result['T_horizon'] = T_horizon

    print('elapsed time:', (result or {}).get('solve_time_total', end - start))
    print(result if result else 'Results dictionary is empty.')

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'CVDP23_finite_time', result if result else {})
