"""
CVDP22_proposed -- adapted Coupled van der Pol benchmark for PRoTECT v2.

ORIGINAL ARCH-COMP 2022 CVDP22 spec (unmodified in this script as a
reference) uses
  x_{1,2}(0) in [1.25, 1.55],  y_{1,2}(0) in [2.35, 2.45]
  unsafe    : y_{1,2} >= 3.7,
  b = 70 fixed.
With these settings PRoTECT v2's infinite-time SOS programme converges
to gamma ~ lambda (no real separation), and the pointwise validator
flags init_slack = -1e-5, unsafe_slack = +2e-5, Lie_slack = +1e-4 --
i.e. the certificate is "coefficient-clean" but pointwise-unsound.

The PROPOSED variant tightens ONLY the initial set:
  x_{1,2}(0) in [1.35, 1.45], y_{1,2}(0) in [2.39, 2.41]
(half the original side lengths in x, fifth in y). The unsafe set,
state space, dynamics, and parameter b = 70 are unchanged.

This gives the SOS programme room to find a barrier with a real
gamma < lambda gap, which the v2 combined coefficient + pointwise
validator certifies as PASS at the strict 1e-8 tolerance.

This variant is proposed as a future-ARCH-COMP target for PRoTECT
demonstrating the v2 pointwise validator on a system that is
*just-out-of-reach* on the original spec but cleanly certifiable on
a tightened spec.
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.solve_helpers import solve_safety_problem
from src.functions.result_export import write_result_json


if __name__ == '__main__':
    dim = 4
    b = 70.0

    # TIGHTENED initial set (only change from CVDP22).
    L_initial = np.array([1.35, 2.39, 1.35, 2.39])
    U_initial = np.array([1.45, 2.41, 1.45, 2.41])

    L_space = np.array([-3.0, -4.0, -3.0, -4.0])
    U_space = np.array([ 3.0,  4.0,  3.0,  4.0])

    L_u1 = L_space.copy(); L_u1[1] = 3.7
    L_u2 = L_space.copy(); L_u2[3] = 3.7
    L_unsafe = np.array([L_u1, L_u2])
    U_unsafe = np.array([U_space.copy(), U_space.copy()])

    x = sp.symbols(f'x0:{dim}')
    mu = 1.0
    f = np.array([
        x[1],
        mu*(1 - x[0]**2)*x[1] + b*(x[2] - x[0]) - x[0],
        x[3],
        mu*(1 - x[2]**2)*x[3] - b*(x[2] - x[0]) - x[2],
    ])
    start = time.time()
    result, solver_used = solve_safety_problem(
        degrees=[2, 4],
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        validate_tolerance=1e-8,
    )
    if result:
        result['solver'] = solver_used
        result['variant'] = 'CVDP22_proposed (tightened init)'
    end = time.time()

    print('elapsed time:', (result or {}).get('solve_time_total', end - start))
    print(result if result else 'Results dictionary is empty.')

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'CVDP22_proposed', result if result else {})
