"""
ROBE25 -- Robertson autocatalytic, 2026 spec.

Source: ARCH-COMP 2026 Nonlinear Dynamics report, Sec. 3.2.

Dynamics (3-D, polynomial, continuous-time):
    x' = -alpha*x + beta*y*z
    y' =  alpha*x - beta*y*z - gamma*y^2
    z' =  gamma*y^2

with alpha = 0.4 fixed and three (beta, gamma) instances:
    1: (1e2, 1e3)
    2: (1e3, 1e5)
    3: (1e3, 1e7)

Initial condition x(0) = 1, y(0) = z(0) = 0.

The paper's specification is the FINAL-TIME WIDTH of x+y+z at t=40s
(< 1e-5), which PRoTECT cannot directly verify. We use the same
positivity-plus-conservation safety reformulation as ROBE21 (rescaled
y to V = scale_y * y) so the SOS programme is well conditioned.
The discussion section notes the specification mismatch.

This is the 2026-spec attempt; if the SOS programme is infeasible at
low degree (which it can be for instances 2 / 3 given the very large
coefficients), the runner falls back to the corresponding ROBE21 row.
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS import ct_DS
from src.functions.figure_export import export_figure
from src.functions.solve_helpers import solve_safety_problem
from src.functions.result_export import write_result_json


# Per-instance parameters and rescaling. The rescaling scale_y is chosen
# to bring the dominant coefficient (gamma / scale_y) to within an order
# of magnitude of the slow rate alpha.
INSTANCES = {
    '1': {'alpha': 0.4, 'beta': 1e2, 'gamma': 1e3, 'scale_y': 1e1, 'eps': 1e-3},
    '2': {'alpha': 0.4, 'beta': 1e3, 'gamma': 1e5, 'scale_y': 1e3, 'eps': 5e-3},
    '3': {'alpha': 0.4, 'beta': 1e3, 'gamma': 1e7, 'scale_y': 1e4, 'eps': 1e-2},
}


if __name__ == '__main__':
    instance = os.environ.get('ROBE25_INSTANCE', '1')
    p = INSTANCES[instance]
    alpha = p['alpha']; beta = p['beta']; gamma_p = p['gamma']
    scale_y = p['scale_y']; eps = p['eps']

    # Rescaled state (u, V, w) = (x, scale_y * y, z); rescaled dynamics:
    #   du/dt = -alpha*u + (beta/scale_y) * V * w
    #   dV/dt =  alpha*scale_y*u - beta*V*w - (gamma/scale_y) * V^2
    #   dw/dt =  (gamma/scale_y^2) * V^2
    a_uw = beta / scale_y
    a_Vw = beta
    a_V2 = gamma_p / scale_y
    a_w  = gamma_p / scale_y**2
    a_uV = alpha * scale_y

    dim = 3

    L_initial = np.array([1.0 - eps, 0.0,             0.0])
    U_initial = np.array([1.0 + eps, scale_y * eps,   eps])

    L_space = np.array([0.0, 0.0,           0.0])
    U_space = np.array([1.1, scale_y * 0.5, 1.1])

    # Same shape of unsafe sets as ROBE21: u below 0.9, or w above 1.0.
    L_unsafe1 = L_space.copy()
    U_unsafe1 = np.array([0.9, U_space[1], U_space[2]])

    L_unsafe2 = np.array([L_space[0], L_space[1], 1.0])
    U_unsafe2 = U_space.copy()

    L_unsafe = np.array([L_unsafe1, L_unsafe2])
    U_unsafe = np.array([U_unsafe1, U_unsafe2])

    x = sp.symbols(f'x0:{dim}')

    f1 = -alpha*x[0] + a_uw * x[1] * x[2]
    f2 =  a_uV*x[0] - a_Vw * x[1] * x[2] - a_V2 * x[1]**2
    f3 =  a_w * x[1]**2
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
    write_result_json(fig_dir, f'ROBE25_{instance}', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, f'ROBE25_{instance}.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=0, dim_y=2,
        title=f'ROBE25 instance {instance} (autocatalytic, 2026 spec)',
        x_label='u', y_label='w',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
