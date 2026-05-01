"""
End-to-end check: ct_DS_robust on the actual CVDP23 dynamics with
b in [1, 3]. Expected: a parameter-independent barrier B(x) found in
substantially less wall time than the lifting-encoding baseline (615.7 s
in the 2026-NLN stretch run).

This is the v2 demonstration that feature 7a unlocks CVDP23/b_unc.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy as sp


if __name__ == '__main__':
    from src.functions.ct_DS_robust import ct_DS_robust

    dim = 4
    x = sp.symbols('x0:4')           # x1, y1, x2, y2
    p = sp.symbols('b0:1')           # uncertain coupling b in [1, 3]

    L_initial = np.array([1.25, 2.35, 1.25, 2.35])
    U_initial = np.array([1.55, 2.45, 1.55, 2.45])

    L_space = np.array([-3.0, -3.0, -3.0, -3.0])
    U_space = np.array([ 3.0,  3.0,  3.0,  3.0])

    # Unsafe 1: y1 >= 2.75
    L_u1 = L_space.copy(); L_u1[1] = 2.75; U_u1 = U_space.copy()
    # Unsafe 2: y2 >= 2.75
    L_u2 = L_space.copy(); L_u2[3] = 2.75; U_u2 = U_space.copy()
    L_unsafe = np.array([L_u1, L_u2])
    U_unsafe = np.array([U_u1, U_u2])

    P_lo = np.array([1.0])
    P_hi = np.array([3.0])

    mu = 1
    f = np.array([
        x[1],
        mu*(1 - x[0]**2)*x[1] + p[0]*(x[2] - x[0]) - x[0],
        x[3],
        mu*(1 - x[2]**2)*x[3] - p[0]*(x[2] - x[0]) - x[2],
    ])

    solver = os.environ.get('PROTECT_SOLVER', 'mosek')

    print(f'Solver: {solver}; lifting baseline: 615.7 s @ degree 4')
    print('Running ct_DS_robust ...', flush=True)
    start = time.time()
    result = None
    for degree in range(2, 7, 2):
        result = ct_DS_robust(
            b_degree=degree, dim=dim,
            L_initial=L_initial, U_initial=U_initial,
            L_unsafe=L_unsafe, U_unsafe=U_unsafe,
            L_space=L_space, U_space=U_space,
            x=x, f=f,
            p_syms=p, P_lo=P_lo, P_hi=P_hi,
            solver=solver,
        )
        if result and 'barrier' in result:
            elapsed = time.time() - start
            print(f'  FOUND at degree {degree} in {elapsed:.2f}s')
            print(f'  gamma = {result["gamma"]:.4g}')
            print(f'  lambda = {result["lambda"]:.4g}')
            # confirm B is parameter-independent
            B = result['barrier']
            free = B.free_symbols if hasattr(B, 'free_symbols') else set()
            assert all(sym not in free for sym in p), \
                f'B contained parameter symbol: {free & set(p)}'
            print(f'  barrier verified parameter-independent')
            sys.exit(0)
        if result and 'error' in result:
            print(f'  Degree {degree} -> {result["error"]}')

    elapsed = time.time() - start
    print(f'NOT FOUND after {elapsed:.2f}s; last result: {result}')
    sys.exit(1)
