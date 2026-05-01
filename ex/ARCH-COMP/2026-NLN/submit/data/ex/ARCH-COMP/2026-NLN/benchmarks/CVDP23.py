"""
CVDP23 -- Coupled van der Pol oscillators, 2026 spec with b FIXED at the
midpoint b = 2 of the paper's uncertainty interval [1, 3].

The b-uncertain version (5-D SOS) lives in CVDP23_uncertain.py; this
b = 2 row is the fixed-parameter reduction that keeps the SOS programme
in 4-D. Per the run manifest, both are attempted; this one is faster.

Source: ARCH-COMP 2026 Nonlinear Dynamics report, Sec. 3.3.

Dynamics (4-D, polynomial, continuous-time):
    x1' = y1
    y1' = mu*(1 - x1^2)*y1 + b*(x2 - x1) - x1
    x2' = y2
    y2' = mu*(1 - x2^2)*y2 - b*(x2 - x1) - x2

with mu = 1, b = 2.
Initial set: x_{1,2}(0) in [1.25, 1.55], y_{1,2}(0) in [2.35, 2.45].
Unsafe set: y1 >= 2.75 OR y2 >= 2.75.
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS import ct_DS
from src.functions.figure_export import export_figure
from src.functions.solve_helpers import solve_safety_problem
from src.functions.result_export import write_result_json


if __name__ == '__main__':
    # We FIX b at the midpoint of [1, 3] rather than carrying b as an extra
    # uncertain state. Carrying b lifts the SOS programme to 5-D with cubic
    # cross-terms in (x_i, b), which OOM'd the Docker run. The midpoint
    # b = 2 is representative; the discussion section notes the parametric
    # uncertainty is not directly captured.
    dim = 4
    b = 2.0

    L_initial = np.array([1.25, 2.35, 1.25, 2.35])
    U_initial = np.array([1.55, 2.45, 1.55, 2.45])

    L_space = np.array([-3.0, -3.0, -3.0, -3.0])
    U_space = np.array([ 3.0,  3.0,  3.0,  3.0])

    # Unsafe 1: y1 >= 2.75
    L_unsafe1 = L_space.copy()
    L_unsafe1[1] = 2.75
    U_unsafe1 = U_space.copy()

    # Unsafe 2: y2 >= 2.75
    L_unsafe2 = L_space.copy()
    L_unsafe2[3] = 2.75
    U_unsafe2 = U_space.copy()

    L_unsafe = np.array([L_unsafe1, L_unsafe2])
    U_unsafe = np.array([U_unsafe1, U_unsafe2])

    x = sp.symbols(f'x0:{dim}')  # x[0]=x1, x[1]=y1, x[2]=x2, x[3]=y2
    mu = 1

    f1 = x[1]
    f2 = mu*(1 - x[0]**2)*x[1] + b*(x[2] - x[0]) - x[0]
    f3 = x[3]
    f4 = mu*(1 - x[2]**2)*x[3] - b*(x[2] - x[0]) - x[2]
    f = np.array([f1, f2, f3, f4])
    start = time.time()
    result, solver_used = solve_safety_problem(
        degrees=range(2, 7, 2),
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        validate_tolerance=1e-3,
    )
    if result:
        result['solver'] = solver_used
    end = time.time()

    print("elapsed time:", (result or {}).get("solve_time_total", end - start))
    print(result if result else "Results dictionary is empty.")

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'CVDP23_b2', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, 'CVDP23.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=0, dim_y=1,
        title='CVDP23 (b=2, fixed-midpoint reduction)',
        x_label='x_{1}', y_label='y_{1}',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
