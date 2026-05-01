"""Tests for src/functions/sets.py (v2 features 2a/2b/2c)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy as sp

from src.functions.sets import (
    polytope_inequalities, sublevel_set, quadratic_form_set,
    union_of_sets, box_to_polytope,
)


def test_polytope_basic():
    x = sp.symbols('x0:2')
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([1, 1, 1, 1])
    polys = polytope_inequalities(x, A, b)
    assert len(polys) == 4
    assert sp.simplify(polys[0] - (1 - x[0])) == 0
    assert sp.simplify(polys[3] - (1 + x[1])) == 0


def test_polytope_shape_check():
    x = sp.symbols('x0:2')
    try:
        polytope_inequalities(x, np.zeros((3, 3)), np.zeros(3))
    except ValueError:
        return
    raise AssertionError("expected ValueError for shape mismatch")


def test_sublevel_set_le():
    x, y = sp.symbols('x y')
    g = x**2 + y**2 - 1   # disk: g <= 0
    polys = sublevel_set(g, sense='le')
    assert sp.simplify(polys[0] - (-g)) == 0


def test_quadratic_form_set_unit_disk():
    x = sp.symbols('x0:2')
    Q = np.eye(2)
    c = np.zeros(2)
    d = -1
    polys = quadratic_form_set(x, Q, c, d, sense='le')
    diff = sp.simplify(polys[0] - (1 - x[0]**2 - x[1]**2))
    assert diff == 0


def test_union_of_sets_validates():
    x = sp.symbols('x0:1')
    region_a = [x[0] - 1]
    region_b = [-x[0] - 1]
    out = union_of_sets([region_a, region_b])
    assert len(out) == 2
    assert sp.simplify(out[0][0] - (x[0] - 1)) == 0


def test_box_to_polytope():
    x = sp.symbols('x0:2')
    polys = box_to_polytope(x, [0, -1], [2, 3])
    # 4 entries: x0 - 0, 2 - x0, x1 + 1, 3 - x1
    assert len(polys) == 4
    assert sp.simplify(polys[1] - (2 - x[0])) == 0


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
