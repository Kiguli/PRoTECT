"""
TRAF22 with ct_DS_robust: q_s, q_c as PARAMETERS not states.

Same dynamics as the existing TRAF22.py, but the sinc auxiliary
variables q_s = sinc(psi) and q_c = sinc(psi/2)^2 are passed to
ct_DS_robust as uncertain parameters. This drops B's search basis from
7-D to 5-D while keeping the soundness of the sinc relaxation.

If this converges in reasonable wall time, migrate the benchmark.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy as sp

from src.functions.ct_DS_robust import ct_DS_robust
from src.functions.sinc_relaxation import sinc_bounds


if __name__ == '__main__':
    l_wb = 2.578
    psi_max = 0.5
    delta_max = 0.4
    sinc_lo, sinc_sq_half_lo = sinc_bounds(psi_max)
    v_nom = 5.0

    # Drop sx (irrelevant to lane-keeping safety) and q_c (only appeared
    # in the sx dynamics). Keeps Lie degree even and reduces state dim.
    dim = 4
    x = sp.symbols('x0:4')
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

    solver = os.environ.get('PROTECT_SOLVER', 'mosek')
    print(f'Solver: {solver}; degrees 2,4,6')

    start = time.time()
    result = None
    for degree in range(2, 7, 2):
        t0 = time.time()
        result = ct_DS_robust(
            b_degree=degree, dim=dim,
            L_initial=L_initial, U_initial=U_initial,
            L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
            L_space=L_space,     U_space=U_space,
            x=x, f=f,
            p_syms=p, P_lo=P_lo, P_hi=P_hi,
            solver=solver,
        )
        dt = time.time() - t0
        if result and 'barrier' in result:
            elapsed = time.time() - start
            print(f'FOUND at degree {degree} in {elapsed:.2f}s (last call {dt:.2f}s)')
            print(f'  gamma  = {result["gamma"]:.4g}')
            print(f'  lambda = {result["lambda"]:.4g}')
            sys.exit(0)
        if result and 'error' in result:
            print(f'  Degree {degree} -> {result["error"]}  ({dt:.2f}s)')

    elapsed = time.time() - start
    print(f'NOT FOUND after {elapsed:.2f}s')
    sys.exit(1)
