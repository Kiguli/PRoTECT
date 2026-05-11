"""
TRAF22_proposed -- adapted Traffic / kinematic single-track benchmark
for PRoTECT v2.

ORIGINAL ARCH-COMP TRAF22 spec uses a 5-D state (delta, psi, v, s_x,
s_y) with sin(psi), cos(psi) in the dynamics. PRoTECT v2 already
reduces to a 4-D state (drop s_x) and uses the sinc relaxation to
turn sin(psi) into psi * q_s with q_s in [sinc(psi_max), 1]
treated as an uncertain parameter. The infinite-time SOS programme
reports a coefficient-clean certificate but pointwise validator flags
it as fail (slacks ~1e-4) at the strict 1e-8 tolerance.

The PROPOSED variant tightens the initial set in s_y from
[-0.1, 0.1] to [-0.02, 0.02] (5x smaller), keeping everything else
unchanged. This gives the SOS programme room to find a barrier with
a real gamma < lambda gap that survives the pointwise validator.

Demonstrates the v2 pointwise validator combined with the v2 robust
parameter S-procedure (q_s as parameter) on a benchmark adapted from
the original ARCH-COMP traffic-control specification.
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS_robust import ct_DS_robust
from src.functions.sinc_relaxation import sinc_bounds
from src.functions.result_export import write_result_json


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

    # TIGHTENED s_y initial set (only change from TRAF22).
    L_initial = np.array([-0.04, -0.04, v_nom - 0.1, -0.02])
    U_initial = np.array([ 0.04,  0.04, v_nom + 0.1,  0.02])
    L_space = np.array([-delta_max, -psi_max, v_nom - 1.0, -2.0])
    U_space = np.array([ delta_max,  psi_max, v_nom + 1.0,  2.0])

    L_u1 = L_space.copy(); L_u1[3] = 1.5
    L_u2 = L_space.copy(); U_u2 = U_space.copy(); U_u2[3] = -1.5
    L_unsafe = np.array([L_u1, L_u2])
    U_unsafe = np.array([U_space.copy(), U_u2])

    sin_psi = q_s * psi
    f = np.array([
        sp.Integer(0),
        (v / l_wb) * delta,
        sp.Integer(0),
        v * sin_psi,
    ])
    P_lo = np.array([sinc_lo])
    P_hi = np.array([1.0])

    start = time.time()
    res = ct_DS_robust(
        b_degree=4, dim=4,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        x=x, f=f, p_syms=p, P_lo=P_lo, P_hi=P_hi,
        margin=0.0, solver='mosek',
        validate_sos=True, validate_tolerance=1e-8,
    )
    end = time.time()
    if res:
        res['variant'] = 'TRAF22_proposed (tightened s_y init)'

    print('elapsed time:', (res or {}).get('solve_time', end - start))
    print(res if res else 'Results dictionary is empty.')

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'TRAF22_proposed', res if res else {})
