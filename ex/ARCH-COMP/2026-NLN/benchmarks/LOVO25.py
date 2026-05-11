"""
LOVO25 -- Lotka-Volterra with tangential crossings (ARCH-COMP NLN benchmark).

Source: ARCH-COMP 2026 Nonlinear Dynamics report, Sec. 3.5.

Dynamics (2-D, polynomial, continuous-time):
    x' = 3*x - 3*x*y
    y' = x*y - y

Cyclic trajectories around the equilibrium (1, 1).

Paper specification (HYBRID): the system is paired with a circular guard
sqrt((x-1)^2 + (y-1)^2) = 0.161 and analysed as a hybrid automaton with
modes {outside, inside} and tangential-crossing-counting properties (no
odd numbers of crossings, etc.).

PRoTECT cannot represent
  - hybrid mode switching with non-axis-aligned guards,
  - crossing-count temporal properties,
  - circular (non-box) unsafe sets directly.

PRoTECT specification (safety reformulation): start from the canonical
initial condition I = (1.3 +/- epsilon, 1.0) with epsilon = 0.012, and
certify that the trajectory remains within an axis-aligned annular region
around the equilibrium (1, 1). Concretely, we verify that the box
    x in [0.6, 1.4] x y in [0.6, 1.4]
is invariant -- this is the box hull from the paper's evaluation figure
(Sec. 3.5.4) and gives a finite-area safety envelope. The crossing-pattern
specifications are documented as out of scope.
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
    dim = 2

    # Initial set: I = (1.3 +/- 0.012, 1.0) per Sec. 3.5.2.
    eps = 0.012
    L_initial = np.array([1.3 - eps, 1.0 - 1e-6])
    U_initial = np.array([1.3 + eps, 1.0 + 1e-6])

    # State space: the box [0.6, 1.4] x [0.6, 1.4] from Fig. 5 of the paper.
    L_space = np.array([0.6, 0.6])
    U_space = np.array([1.4, 1.4])

    # Unsafe sets: the four box-shells outside the safe region. Since
    # PRoTECT requires unsafe regions inside L_space/U_space, we enclose
    # the dynamics in a slightly larger ambient box and place the unsafe
    # sets just outside [0.6, 1.4] x [0.6, 1.4].
    L_ambient = np.array([0.5, 0.5])
    U_ambient = np.array([1.5, 1.5])

    # x <= 0.6
    L_u1 = L_ambient.copy(); U_u1 = np.array([0.6, U_ambient[1]])
    # x >= 1.4
    L_u2 = np.array([1.4, L_ambient[1]]); U_u2 = U_ambient.copy()
    # y <= 0.6
    L_u3 = L_ambient.copy(); U_u3 = np.array([U_ambient[0], 0.6])
    # y >= 1.4
    L_u4 = np.array([L_ambient[0], 1.4]); U_u4 = U_ambient.copy()

    L_unsafe = np.array([L_u1, L_u2, L_u3, L_u4])
    U_unsafe = np.array([U_u1, U_u2, U_u3, U_u4])

    # Use the ambient box as the SOS state space.
    L_space = L_ambient
    U_space = U_ambient

    x = sp.symbols(f'x0:{dim}')
    f1 = 3*x[0] - 3*x[0]*x[1]
    f2 = x[0]*x[1] - x[1]
    f = np.array([f1, f2])
    start = time.time()
    result, solver_used = solve_safety_problem(
        degrees=range(2, 7, 2),
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        validate_tolerance=1e-8,
    )
    if result:
        result['solver'] = solver_used
    end = time.time()

    print("elapsed time:", (result or {}).get("solve_time_total", end - start))
    print(result if result else "Results dictionary is empty.")

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'LOVO25', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, 'LOVO25.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=0, dim_y=1,
        title='LOVO25',
        x_label='x', y_label='y',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
