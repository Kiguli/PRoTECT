"""
ROBE21 -- Robertson chemical reaction, ORIGINAL ARCH-COMP spec.

Original Robertson kinetics (very stiff):
    dx/dt = -0.04*x + 1e4*y*z
    dy/dt =  0.04*x - 1e4*y*z - 3e7*y^2
    dz/dt =  3e7*y^2

Rescaling y -> V = 1e4 * y compresses the coefficient range from 9 orders
of magnitude (0.04 ... 3e7) to 5 orders, which is what the prior PRoTECT
submission used:
    du/dt = -0.04*u + V*w
    dV/dt =  400*u  - 1e4*V*w - 300*V^2
    dw/dt =  0.3*V^2

Initial condition x = 1, y = z = 0 expanded by an instance-dependent
half-width epsilon. Three instances (selected via env ROBE21_INSTANCE):
    1: epsilon = 0.001
    2: epsilon = 0.005
    3: epsilon = 0.01

Unsafe sets:
    (a) u drops below 0.9    (loss of x species)
    (b) w exceeds 1.0        (over-accumulation of z species)

This row matches the canonical results.csv from the prior PRoTECT
submission (137.5 / 224.0 / 320.9 s for instances 1 / 2 / 3).
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS import ct_DS
from src.functions.figure_export import export_figure
from src.functions.solve_helpers import solve_safety_problem
from src.functions.result_export import write_result_json


INSTANCES = {
    '1': {'eps': 0.001},
    '2': {'eps': 0.005},
    '3': {'eps': 0.01},
}

SCALE_Y = 1e4


if __name__ == '__main__':
    instance = os.environ.get('ROBE21_INSTANCE', '1')
    eps = INSTANCES[instance]['eps']

    dim = 3

    L_initial = np.array([1.0 - eps,   0.0,           0.0])
    U_initial = np.array([1.0 + eps,   SCALE_Y * eps, eps])

    L_space = np.array([0.0, 0.0,             0.0])
    U_space = np.array([1.1, SCALE_Y * 0.01,  1.1])

    L_unsafe1 = L_space.copy()
    U_unsafe1 = np.array([0.9, U_space[1], U_space[2]])

    L_unsafe2 = np.array([L_space[0], L_space[1], 1.0])
    U_unsafe2 = U_space.copy()

    L_unsafe = np.array([L_unsafe1, L_unsafe2])
    U_unsafe = np.array([U_unsafe1, U_unsafe2])

    x = sp.symbols(f'x0:{dim}')   # x[0]=u, x[1]=V (= 1e4*y), x[2]=w (= z)

    f1 = -0.04*x[0] + x[1]*x[2]
    f2 =  400*x[0]  - 1e4*x[1]*x[2] - 300*x[1]**2
    f3 =  0.3*x[1]**2
    f = np.array([f1, f2, f3])
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
    write_result_json(fig_dir, f'ROBE21_{instance}', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, f'ROBE21_{instance}.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=0, dim_y=2,
        title=f'ROBE21 instance {instance} (original spec)',
        x_label='u', y_label='w',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
