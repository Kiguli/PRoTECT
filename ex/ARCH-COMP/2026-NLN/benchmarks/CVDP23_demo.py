"""
CVDP23 demonstration variant -- a simplification (in the same spirit as
KeYmaera X's simplification to b=1, m=1, t in [0, 0.1]) that fixes b at
a single value rather than the paper's uncertain interval. The initial
set, unsafe set, and state space all match the paper Eq. (1) Sec. 3.3
EXACTLY -- only the parameter b is fixed.

Differences from the paper's CVDP23:

  * b is FIXED at b = 1.0 (paper specifies b in [1, 3] uncertain).
  * Everything else matches: initial set x_{1,2}(0) in [1.25, 1.55],
    y_{1,2}(0) in [2.35, 2.45]; unsafe y_{1,2} >= 2.75; dynamics
    x_i' = y_i, y_i' = mu (1 - x_i^2) y_i +/- b (x_2 - x_1) - x_i,
    mu = 1.

This is a separate benchmark target -- it does NOT replace the original
CVDP23 reports (those remain on the paper specification).
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

    L_initial = np.array([1.25, 2.35, 1.25, 2.35])
    U_initial = np.array([1.55, 2.45, 1.55, 2.45])

    L_space = np.array([-3.0, -3.0, -3.0, -3.0])
    U_space = np.array([ 3.0,  3.0,  3.0,  3.0])

    # Unsafe: y1 >= 2.75 OR y2 >= 2.75 (paper spec, unchanged).
    unsafe_y = 2.75
    L_u1 = L_space.copy(); L_u1[1] = unsafe_y
    L_u2 = L_space.copy(); L_u2[3] = unsafe_y
    L_unsafe = np.array([L_u1, L_u2])
    U_unsafe = np.array([U_space.copy(), U_space.copy()])

    mu = 1.0
    b = 1.0
    f = np.array([
        x[1],
        mu * (1 - x[0]**2) * x[1] + b * (x[2] - x[0]) - x[0],
        x[3],
        mu * (1 - x[2]**2) * x[3] - b * (x[2] - x[0]) - x[2],
    ])
    start = time.time()
    result, solver_used = solve_safety_problem(
        degrees=range(2, 7, 2),
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        validate_tolerance=1e-8,
        # No robust parameter; b is fixed.
        # No margin requirement -- let the solver give the natural gap.
    )
    if result:
        result['solver'] = solver_used
    end = time.time()

    print('elapsed time:', (result or {}).get('solve_time_total', end - start))
    print(result if result else 'Results dictionary is empty.')

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'CVDP23_demo', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, 'CVDP23_demo.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=0, dim_y=1,
        title=f'CVDP23 demo (b={b}, paper spec)',
        x_label='x_{1}', y_label='y_{1}',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
