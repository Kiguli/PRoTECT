"""
Smoke test for ct_DS_robust (PRoTECT v2 feature 7a).

Tiny 2-D damped oscillator with a single uncertain damping parameter:
    x1' = x2
    x2' = -x1 - p * x2,    p in [0.5, 1.5]

Initial: small box near origin. Unsafe: |x1| > 1.5. State space: |x_i| <= 2.
The system is stable for any p >= 0, so a quadratic barrier should exist
and be valid for ALL p in the parameter box. We check that the search
returns a barrier without parameter dependence (B is over x only).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy as sp


def test_ct_DS_robust_finds_barrier():
    from src.functions.ct_DS_robust import ct_DS_robust

    dim = 2
    x = sp.symbols('x0:2')
    p = sp.symbols('p0:1')

    L_initial = np.array([-0.05, -0.05])
    U_initial = np.array([ 0.05,  0.05])

    L_space = np.array([-2.0, -2.0])
    U_space = np.array([ 2.0,  2.0])

    # Unsafe set 1: x1 >= 1.5
    L_u1 = L_space.copy(); L_u1[0] = 1.5; U_u1 = U_space.copy()
    # Unsafe set 2: x1 <= -1.5
    L_u2 = L_space.copy(); U_u2 = U_space.copy(); U_u2[0] = -1.5

    L_unsafe = np.array([L_u1, L_u2])
    U_unsafe = np.array([U_u1, U_u2])

    P_lo = np.array([0.5])
    P_hi = np.array([1.5])

    f = np.array([
        x[1],
        -x[0] - p[0] * x[1],
    ])

    solver = os.environ.get('PROTECT_SOLVER', 'mosek')

    result = ct_DS_robust(
        b_degree=2, dim=dim,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe, U_unsafe=U_unsafe,
        L_space=L_space, U_space=U_space,
        x=x, f=f,
        p_syms=p, P_lo=P_lo, P_hi=P_hi,
        solver=solver,
    )

    print('result keys:', sorted(result.keys()))
    if 'error' in result:
        raise AssertionError(f'Expected barrier but got error: {result["error"]}')

    assert 'barrier' in result, f'No barrier in result: {result}'

    # Confirm B is over x only (does not contain p symbols).
    B = result['barrier']
    free = B.free_symbols if hasattr(B, 'free_symbols') else set()
    assert all(sym not in free for sym in p), \
        f'Barrier should be parameter-independent but contains: {free & set(p)}'

    print(f'PASS: barrier found at degree 2')
    print(f'      gamma = {result["gamma"]:.4g}')
    print(f'      lambda = {result["lambda"]:.4g}')


if __name__ == '__main__':
    test_ct_DS_robust_finds_barrier()
    print('OK')
