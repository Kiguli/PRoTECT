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

This driver is configurable via environment variables so each individual
(degree, time_orders) attempt can be invoked separately with an external
time-box (since the SOS programme is 6-D and easily exceeds 15 min on
larger sweeps):

    PROTECT_FT_FIX_B = '1' to fix b = 1.0 (drops the parameter-box
                       S-procedure -> 5-D SOS instead of 6-D)
    PROTECT_FT_FIX_B_VAL = float value of fixed b (default 1.0)
    PROTECT_FT_DEGREE = single integer (B_k spatial degree)
    PROTECT_FT_TORDER = single integer (max power of t)
    PROTECT_FT_LABEL  = label suffix appended to the result-JSON filename

When run without these env vars, defaults to the original full sweep.
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

    fix_b = os.environ.get('PROTECT_FT_FIX_B', '0') == '1'
    if fix_b:
        b_val = float(os.environ.get('PROTECT_FT_FIX_B_VAL', '1.0'))
        p = ()
        P_lo = ()
        P_hi = ()
        f = np.array([
            x[1],
            (1 - x[0]**2) * x[1] + b_val * (x[2] - x[0]) - x[0],
            x[3],
            (1 - x[2]**2) * x[3] - b_val * (x[2] - x[0]) - x[2],
        ])
        variant_tag = f'fixedB{b_val}'
    else:
        p = sp.symbols('b0:1')              # uncertain b
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
        variant_tag = 'uncertainB'

    # Allow per-attempt selection of (degree, time_order). If unset, use
    # smallest combination first.
    deg_env = os.environ.get('PROTECT_FT_DEGREE')
    tord_env = os.environ.get('PROTECT_FT_TORDER')
    if deg_env is not None and tord_env is not None:
        degrees = [int(deg_env)]
        time_orders = [int(tord_env)]
    else:
        # One-shot: (degree=2, time_orders=2) is the smallest feasible
        # combination (the dynamics has cubic terms, so the Lie polynomial
        # has total degree 2 + 1 = 3 with time_orders=1, which makes the
        # SOS constraint odd-degree and `add_sos_constraint` rejects it).
        # We found at (2, 2) the certificate exists and is pointwise sound.
        degrees = [2]
        time_orders = [2]

    T_horizon = 7.0
    # Allow run_benchmarks.py to set a FULL label (matching its csv_label
    # convention `<benchmark_id>_<instance>`); fall back to the default
    # 'CVDP23_finite_time' label when no env var is set.
    label = os.environ.get('PROTECT_FT_LABEL', 'CVDP23_finite_time')

    print(f'[CVDP23_finite_time] variant={variant_tag} '
          f'degrees={degrees} time_orders={time_orders} '
          f'T_horizon={T_horizon} label={label}')

    start = time.time()
    result, solver_used = solve_finite_time_safety_problem(
        degrees=degrees,
        time_orders=time_orders,
        T_horizon=T_horizon,
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        p_syms=p, P_lo=P_lo, P_hi=P_hi,
        margin=0.0,
        validate_tolerance=1e-8,
    )
    end = time.time()
    if result:
        result['solver'] = solver_used
        result['T_horizon'] = T_horizon
        result['variant'] = variant_tag

    elapsed = (result or {}).get('solve_time_total', end - start)
    print(f'[CVDP23_finite_time] elapsed time: {elapsed:.2f}s')
    if result:
        for k, v in result.items():
            if k == 'barrier':
                continue
            print(f'  {k}: {v}')
    else:
        print('Results dictionary is empty.')

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, label, result if result else {})
