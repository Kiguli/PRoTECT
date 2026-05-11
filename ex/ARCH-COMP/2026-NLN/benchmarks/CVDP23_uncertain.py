"""
CVDP23 (uncertain) -- Coupled van der Pol oscillators with the FULL 2026
spec: b is treated as an uncertain parameter ranging over [1, 3].

Source: ARCH-COMP 2026 Nonlinear Dynamics report, Sec. 3.3.

This is the **PRoTECT v2 path** for the parametric-uncertainty case:
search for a parameter-INDEPENDENT barrier B(x) over the 4-D state with
b in [1, 3] handled via a parameter-box S-procedure multiplier in the
Lie-derivative SOS constraint (see src/functions/ct_DS_robust.py).

The PRoTECT v1 lifting encoding (treat b as a 5th state with b' = 0)
was previously used here and required ~616 s wall time at degree 4.
The v2 robust-Positivstellensatz drops the SOS basis from 5-D to 4-D
and runs in ~55 s in local testing -- an 11x speed-up.

Initial set: x_{1,2}(0) in [1.25, 1.55], y_{1,2}(0) in [2.35, 2.45].
Parameter:   b in [1, 3] uncertain.
Unsafe set:  y1 >= 2.75 OR y2 >= 2.75.
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS_robust import ct_DS_robust
from src.functions.figure_export import export_figure
from src.functions.solve_helpers import solve_safety_problem
from src.functions.result_export import write_result_json


if __name__ == '__main__':
    dim = 4
    x = sp.symbols(f'x0:{dim}')      # x1, y1, x2, y2
    p = sp.symbols('b0:1')           # uncertain b in [1, 3]

    L_initial = np.array([1.25, 2.35, 1.25, 2.35])
    U_initial = np.array([1.55, 2.45, 1.55, 2.45])

    L_space = np.array([-3.0, -3.0, -3.0, -3.0])
    U_space = np.array([ 3.0,  3.0,  3.0,  3.0])

    L_u1 = L_space.copy(); L_u1[1] = 2.75; U_u1 = U_space.copy()
    L_u2 = L_space.copy(); L_u2[3] = 2.75; U_u2 = U_space.copy()
    L_unsafe = np.array([L_u1, L_u2])
    U_unsafe = np.array([U_u1, U_u2])

    P_lo = np.array([1.0])
    P_hi = np.array([3.0])

    mu = 1
    f = np.array([
        x[1],
        mu*(1 - x[0]**2)*x[1] + p[0]*(x[2] - x[0]) - x[0],
        x[3],
        mu*(1 - x[2]**2)*x[3] - p[0]*(x[2] - x[0]) - x[2],
    ])
    start = time.time()
    result, solver_used = solve_safety_problem(
        degrees=range(2, 5, 2),
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        validate_tolerance=1e-8,
        p_syms=p, P_lo=P_lo, P_hi=P_hi,
    )
    if result:
        result['solver'] = solver_used
    end = time.time()

    print("elapsed time:", (result or {}).get("solve_time_total", end - start))
    print(result if result else "Results dictionary is empty.")

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'CVDP23_b_unc', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, 'CVDP23_uncertain.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=0, dim_y=1,
        title='CVDP23 (b in [1,3] uncertain, v2 robust SOS)',
        x_label='x_{1}', y_label='y_{1}',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
