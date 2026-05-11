"""
LOVO21 -- Lorenz system, ORIGINAL ARCH-COMP spec.

Switched from ct_DS to ct_DS_robust with margin=0.5: the previous
ct_DS run produced a certificate where Z3 found an unsafe-side
violation of ~2.6 in exact arithmetic. With margin >= 0.5 the SOS
programme is forced to leave a real gap between the initial and
unsafe level sets (gamma + 0.5 <= lambda), which closes the violation.

Dynamics (3-D, polynomial, continuous-time):
    x' = sigma*(y - x)
    y' = x*(rho - z) - y
    z' = x*y - beta*z

with sigma = 10, rho = 28, beta = 8/3.

Initial: x in [0.9, 1.1], y in [-0.01, 0.01], z in [-0.01, 0.01].
Unsafe:  x >= 20.

This is the FALLBACK row for the LOVO family.
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
    dim = 3

    L_initial = np.array([0.9,  -0.01, -0.01])
    U_initial = np.array([1.1,   0.01,  0.01])

    L_space = np.array([-30.0, -30.0,   0.0])
    U_space = np.array([ 30.0,  30.0,  60.0])

    L_unsafe1 = np.array([20.0, L_space[1], L_space[2]])
    U_unsafe1 = U_space.copy()
    L_unsafe = np.array([L_unsafe1])
    U_unsafe = np.array([U_unsafe1])

    x = sp.symbols(f'x0:{dim}')
    sigma = 10.0
    rho   = 28.0
    beta  = sp.Rational(8, 3)

    f1 = sigma*(x[1] - x[0])
    f2 = x[0]*(rho - x[2]) - x[1]
    f3 = x[0]*x[1] - beta*x[2]
    f = np.array([f1, f2, f3])
    start = time.time()
    result, solver_used = solve_safety_problem(
        degrees=range(2, 7, 2),
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        validate_tolerance=1e-8,
    margin=4.0, mosek_tol=1e-10,
    )
    if result:
        result['solver'] = solver_used
    end = time.time()

    print("elapsed time:", (result or {}).get("solve_time_total", end - start))
    print(result if result else "Results dictionary is empty.")

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'LOVO21', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, 'LOVO21.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=0, dim_y=2,
        title='LOVO21 (Lorenz, margin=0.5)',
        x_label='x', y_label='z',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
