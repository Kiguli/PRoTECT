"""
Smoke tests for the v2 relaxation registry.

Each test verifies the polynomial substitute and the auxiliary box bounds
produced by the registry. We do NOT exercise the SOS pipeline here -- the
goal is to lock down the registry's contract so deeper integrations (e.g.
ct_DS plumbing for the equality multipliers) can rely on it.
"""

import math

import sympy as sp

from src.functions.relaxations import (
    relax,
    relax_sin, relax_cos, relax_tan,
    relax_exp, relax_log,
    relax_sqrt, relax_inv_power,
    relax_sin_cos_pair, stack_relaxations,
)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_relax_sin_basic():
    a = sp.Symbol('a')
    q = sp.Symbol('q')
    r = relax_sin(a, q, math.pi / 4)
    assert r.expr == q * a
    assert len(r.aux_vars) == 1
    sym, lo, hi = r.aux_vars[0]
    assert sym == q
    assert approx(lo, math.sin(math.pi / 4) / (math.pi / 4))
    assert hi == 1.0
    assert r.equalities == []


def test_relax_cos_basic():
    a = sp.Symbol('a')
    r = sp.Symbol('r')
    rel = relax_cos(a, r, math.pi / 4)
    assert sp.simplify(rel.expr - (1 - r * a**2 / 2)) == 0
    sym, lo, hi = rel.aux_vars[0]
    assert sym == r
    half = math.pi / 8
    assert approx(lo, (math.sin(half) / half) ** 2)
    assert hi == 1.0


def test_relax_tan_bounds_at_quarter_pi():
    a = sp.Symbol('a')
    q = sp.Symbol('q')
    rel = relax_tan(a, q, math.pi / 4)
    assert rel.expr == q * a
    _, lo, hi = rel.aux_vars[0]
    assert lo == 1.0
    assert approx(hi, math.tan(math.pi / 4) / (math.pi / 4))


def test_relax_tan_rejects_quarter_pi_or_more():
    a = sp.Symbol('a')
    q = sp.Symbol('q')
    try:
        relax_tan(a, q, math.pi / 2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for angle_max >= pi/2")


def test_relax_exp():
    arg = sp.Symbol('arg')
    q = sp.Symbol('q')
    rel = relax_exp(arg, q, -1.0, 2.0)
    assert rel.expr == q
    _, lo, hi = rel.aux_vars[0]
    assert approx(lo, math.exp(-1))
    assert approx(hi, math.exp(2))


def test_relax_log_rejects_nonpositive_lo():
    arg = sp.Symbol('arg')
    q = sp.Symbol('q')
    try:
        relax_log(arg, q, 0.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for arg_lo <= 0")


def test_relax_sqrt_emits_equality():
    arg = sp.Symbol('arg')
    q = sp.Symbol('q')
    rel = relax_sqrt(arg, q, 1.0, 4.0)
    _, lo, hi = rel.aux_vars[0]
    assert approx(lo, 1.0)
    assert approx(hi, 2.0)
    assert len(rel.equalities) == 1
    assert sp.simplify(rel.equalities[0] - (q**2 - arg)) == 0


def test_relax_inv_power_k3():
    arg = sp.Symbol('arg')
    q = sp.Symbol('q')
    rel = relax_inv_power(arg, q, 1.0, 2.0, k=3)
    _, lo, hi = rel.aux_vars[0]
    assert approx(lo, 1.0 / 8.0)
    assert approx(hi, 1.0)
    assert sp.simplify(rel.equalities[0] - (q * arg**3 - 1)) == 0


def test_registry_dispatch():
    a = sp.Symbol('a')
    q = sp.Symbol('q')
    rel = relax('sin', a, q, math.pi / 6)
    assert rel.expr == q * a


def test_registry_unknown_name():
    try:
        relax('bogus')
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown relaxation name")


def test_relax_sin_cos_pair_shares_angle():
    psi = sp.Symbol('psi')
    qs = sp.Symbol('qs')
    qc = sp.Symbol('qc')
    sin_rel, cos_rel = relax_sin_cos_pair(psi, qs, qc, math.pi / 4)
    assert sin_rel.expr == qs * psi
    assert sp.simplify(cos_rel.expr - (1 - qc * psi**2 / 2)) == 0


def test_stack_relaxations():
    a = sp.Symbol('a')
    q1 = sp.Symbol('q1')
    q2 = sp.Symbol('q2')
    aux, eqs = stack_relaxations(
        relax_sin(a, q1, math.pi / 4),
        relax_inv_power(a + 1, q2, 0.5, 1.0, k=2),
    )
    assert len(aux) == 2
    assert len(eqs) == 1
    assert sp.simplify(eqs[0] - (q2 * (a + 1)**2 - 1)) == 0


if __name__ == '__main__':
    # Tiny in-file runner: invoke each test_* function and report pass/fail.
    import sys
    g = dict(globals())
    failed = 0
    for name, fn in sorted(g.items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as exc:
                failed += 1
                print(f'  FAIL {name}: {exc}')
    print('---')
    print(f'{failed} failure(s)')
    sys.exit(1 if failed else 0)
