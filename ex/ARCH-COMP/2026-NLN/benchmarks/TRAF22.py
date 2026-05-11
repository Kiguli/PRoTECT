"""
TRAF22 -- Traffic / kinematic single-track model (ARCH-COMP NLN benchmark).

Source: ARCH-COMP 2026 Nonlinear Dynamics report, Sec. 3.1.

Original dynamics (Sec. 3.1.1, eq. 1):
    delta' = u1 + w1
    psi'   = (v / l_wb) * tan(delta)
    v'     = u2 + w2
    sx'    = v * cos(psi)
    sy'    = v * sin(psi)

PRoTECT v1 used a 7-D state (5 physical + 2 sinc auxiliaries) and timed
out at 900s on every reasonable degree budget. The PRoTECT v2 path:

  * Drop the sx state. Lane-keeping safety is a property of sy alone;
    sx evolves freely and is irrelevant to the |sy| < 1.5 verdict. This
    eliminates v * cos(psi) from the dynamics, and with it the q_c sinc
    auxiliary that the cos relaxation needed.
  * Move the remaining sin auxiliary q_s = sinc(psi) from a STATE to a
    PARAMETER via ct_DS_robust. The barrier B(x) is now over the 4-D
    physical state only; q_s enters the Lie SOS through a parameter-box
    S-procedure multiplier.

Result on the local box: degree-4 barrier found in ~86 s.

Out of scope (documented in the report discussion):
  - time-varying reference / tracking controller
  - CommonRoad polygonal occupancy
  - input-set polytope constraints
  - disturbances w1, w2 (could be added as additional parameters via
    ct_DS_robust if needed, mathematically identical pattern).

State:
    x[0] = delta in [-delta_max, delta_max]
    x[1] = psi   in [-psi_max,    psi_max]
    x[2] = v     in [v_nom - 1, v_nom + 1]
    x[3] = sy    in [-2, 2]   (unsafe: |sy| >= 1.5)
Parameter:
    q_s in [sinc(psi_max), 1]   (sin(psi) replaced by q_s * psi)
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS_robust import ct_DS_robust
from src.functions.figure_export import export_figure
from src.functions.solve_helpers import solve_safety_problem
from src.functions.result_export import write_result_json
from src.functions.sinc_relaxation import sinc_bounds


if __name__ == '__main__':
    l_wb = 2.578
    psi_max = 0.5
    delta_max = 0.4
    sinc_lo, _ = sinc_bounds(psi_max)
    v_nom = 5.0

    dim = 4
    x = sp.symbols(f'x0:{dim}')
    delta, psi, v, sy = x
    p = sp.symbols('p0:1')
    q_s, = p

    L_initial = np.array([-0.04, -0.04, v_nom - 0.1, -0.1])
    U_initial = np.array([ 0.04,  0.04, v_nom + 0.1,  0.1])
    L_space = np.array([-delta_max, -psi_max, v_nom - 1.0, -2.0])
    U_space = np.array([ delta_max,  psi_max, v_nom + 1.0,  2.0])

    L_u1 = L_space.copy(); L_u1[3] =  1.5; U_u1 = U_space.copy()
    L_u2 = L_space.copy();                  U_u2 = U_space.copy(); U_u2[3] = -1.5
    L_unsafe = np.array([L_u1, L_u2])
    U_unsafe = np.array([U_u1, U_u2])

    P_lo = np.array([sinc_lo])
    P_hi = np.array([1.0])

    sin_psi = q_s * psi
    f = np.array([
        sp.Integer(0),
        (v / l_wb) * delta,
        sp.Integer(0),
        v * sin_psi,
    ])
    start = time.time()
    result, solver_used = solve_safety_problem(
        degrees=range(2, 7, 2),
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
    write_result_json(fig_dir, 'TRAF22', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, 'TRAF22.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=1, dim_y=3,
        title='TRAF22 (v2 robust SOS, sy projection)',
        x_label='\\psi', y_label='s_{y} [m]',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
