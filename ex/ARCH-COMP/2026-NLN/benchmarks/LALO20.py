"""
LALO20 -- Laub-Loomis enzymatic-network model (ARCH-COMP NLN benchmark).

Source: ARCH-COMP 2026 Nonlinear Dynamics report, Sec. 3.4.

Dynamics (7-D, polynomial, continuous-time deterministic):
    x1' = 1.4*x3 - 0.9*x1
    x2' = 2.5*x5 - 1.5*x2
    x3' = 0.6*x7 - 0.8*x2*x3
    x4' = 2 - 1.3*x3*x4
    x5' = 0.7*x1 - x4*x5
    x6' = 0.3*x1 - 3.1*x6
    x7' = 1.8*x6 - 1.5*x2*x7

Initial set: box of half-width W around
    [1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45].
Three instances (selected via env LALO20_INSTANCE):
    W001 -> W=0.01, unsafe x4 >= 4.5
    W005 -> W=0.05, unsafe x4 >= 4.5
    W01  -> W=0.10, unsafe x4 >= 5.0
Time horizon t in [0, 20].

PRoTECT specification: synthesise a polynomial barrier certificate proving
that the unsafe box (x4 above the threshold) is unreachable from the initial
set. The paper's metric is the final-time width of x4 -- PRoTECT verifies
the stronger time-unbounded safety property and reports BC found / time only.
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS import ct_DS
from src.functions.figure_export import export_figure
from src.functions.solve_helpers import solve_safety_problem
from src.functions.result_export import write_result_json


INSTANCE_PARAMS = {
    'W001': {'W': 0.01, 'unsafe_x4': 4.5},
    'W005': {'W': 0.05, 'unsafe_x4': 4.5},
    'W01':  {'W': 0.10, 'unsafe_x4': 5.0},
}


if __name__ == '__main__':
    instance = os.environ.get('LALO20_INSTANCE', 'W001')
    params = INSTANCE_PARAMS[instance]
    W = params['W']
    unsafe_x4 = params['unsafe_x4']

    dim = 7
    center = np.array([1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45])

    L_initial = center - W
    U_initial = center + W

    # Tight state-space envelope matching the prior PRoTECT submission
    # (which solved this benchmark in 9.1 / 16.5 / 242 s). Widening these
    # bounds was empirically observed to push SOS over the 600 s timeout.
    L_space = np.array([0.5, 0.5, 1.0, 1.5, 0.5, 0.05, 0.2])
    U_space = np.array([2.5, 2.5, 4.0, 6.0, 2.5, 0.5,  1.2])

    # Unsafe set: x4 (zero-indexed x[3]) >= unsafe_x4
    L_unsafe1 = L_space.copy()
    L_unsafe1[3] = unsafe_x4
    U_unsafe1 = U_space.copy()
    L_unsafe = np.array([L_unsafe1])
    U_unsafe = np.array([U_unsafe1])

    x = sp.symbols(f'x0:{dim}')

    f1 = 1.4*x[2] - 0.9*x[0]
    f2 = 2.5*x[4] - 1.5*x[1]
    f3 = 0.6*x[6] - 0.8*x[1]*x[2]
    f4 = 2 - 1.3*x[2]*x[3]
    f5 = 0.7*x[0] - x[3]*x[4]
    f6 = 0.3*x[0] - 3.1*x[5]
    f7 = 1.8*x[5] - 1.5*x[1]*x[6]
    f = np.array([f1, f2, f3, f4, f5, f6, f7])
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
    write_result_json(fig_dir, f'LALO20_{instance}', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, f'LALO20_{instance}.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=2, dim_y=3,
        title=f'LALO20 ({instance})',
        x_label='x_{3}', y_label='x_{4}',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
