"""Tests for src/functions/pade.py (v2 feature 1b)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp

from src.functions.pade import (
    pade_coefficients, pade_expression, polynomialise_via_pade,
)


def test_pade_exp_2_2():
    """Pade [2/2] of exp at 0 is (1 + x/2 + x^2/12) / (1 - x/2 + x^2/12)."""
    x = sp.Symbol('x')
    P, Q = pade_expression(sp.exp(x), x, 2, 2)
    expected_P = 1 + x/2 + x**2/12
    expected_Q = 1 - x/2 + x**2/12
    assert sp.simplify(P - expected_P) == 0
    assert sp.simplify(Q - expected_Q) == 0


def test_pade_coefficients_q0_is_1():
    # 1/(1-x) = 1 + x + x^2 + ...; Pade [0/1] should be 1/(1-x).
    coeffs = [1, 1, 1, 1]
    p, q = pade_coefficients(coeffs, 0, 1)
    assert q[0] == 1
    assert sp.simplify(q[1] - (-1)) == 0
    assert p[0] == 1


def test_polynomialise_via_pade_returns_equality():
    x = sp.Symbol('x')
    q = sp.Symbol('q')
    sub_expr, eq, P, Q = polynomialise_via_pade(sp.cos(x), x, q, 2, 2)
    assert sub_expr == q
    # eq should be q*Q - P
    assert sp.simplify(eq - (q*Q - P)) == 0


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print(f'  PASS {name}')
            except Exception as e:
                failed += 1; print(f'  FAIL {name}: {e}')
    print(f'{failed} failure(s)')
    sys.exit(1 if failed else 0)
