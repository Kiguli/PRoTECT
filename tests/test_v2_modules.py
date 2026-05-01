"""
Smoke tests for v2 modules: vertex enumeration, finite-time barrier,
reach-avoid, hybrid, DAE elimination, slow-fast, block decomposition,
stochastic extensions, NN control. These verify the API contracts
without exercising the full SOS solver.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy as sp


def test_enumerate_vertices_2d():
    from src.functions.vertex_enumeration import enumerate_vertices
    verts = enumerate_vertices([(0.5, 1.5), (1.0, 3.0)])
    assert len(verts) == 4
    assert (0.5, 1.0) in verts
    assert (1.5, 3.0) in verts


def test_intersect_level_sets_picks_max_gamma():
    from src.functions.vertex_enumeration import intersect_level_sets
    out = intersect_level_sets([
        {'barrier': sp.Symbol('B1'), 'gamma': 0.3},
        {'barrier': sp.Symbol('B2'), 'gamma': 0.7},
    ])
    assert out['gamma_max'] == 0.7
    assert out['n_vertices'] == 2


def test_finite_time_barrier_shape():
    from src.functions.finite_time import (
        time_polynomial_barrier, time_box_polynomial, lie_derivative_time_dep,
    )
    x = sp.symbols('x0:2')
    t = sp.Symbol('t')
    B, Bk = time_polynomial_barrier(x, t, degree=2, time_orders=2)
    assert len(Bk) == 3
    g_t = time_box_polynomial(t, 5)
    assert sp.simplify(g_t - t * (5 - t)) == 0
    f = [x[1], -x[0]]
    Lie = lie_derivative_time_dep(B, x, t, f)
    assert t in Lie.free_symbols or t == 0


def test_reach_avoid_conditions():
    from src.functions.reach_avoid import reach_avoid_conditions
    x = sp.symbols('x0:2')
    B = x[0]**2 + x[1]**2
    f = [-x[0], -x[1]]
    out = reach_avoid_conditions(B, x, f, gamma_safe=1.0, gamma_target=0.1,
                                 initial_set_polys=[], unsafe_set_polys=[],
                                 target_set_polys=[])
    assert 'initial' in out and 'avoid' in out and 'reach' in out and 'descent' in out
    # descent = -<grad B, f> - eps = -(-2x0^2 - 2x1^2) - eps = 2(x0^2 + x1^2) - eps
    descent = sp.simplify(out['descent'] - (2*(x[0]**2 + x[1]**2) - 1e-3))
    assert descent == 0


def test_hybrid_mode_lie_and_reset():
    from src.functions.hybrid import Mode, Edge, per_mode_lie_derivative, reset_inclusion_expression
    x = sp.symbols('x0:2')
    f1 = [-x[0], -x[1]]
    m1 = Mode('m1', x, f1, invariant_polys=[], initial_polys=[])
    m2 = Mode('m2', x, [-2*x[0], -2*x[1]], invariant_polys=[])
    B1 = x[0]**2 + x[1]**2
    B2 = (x[0] + 1)**2 + (x[1] + 1)**2
    e = Edge(m1, m2, guard_polys=[], reset_map=lambda src_x: [src_x[0] - 1, src_x[1] - 1])
    expr = reset_inclusion_expression(B1, B2, e)
    # B2 at reset(x) = (x0 - 1 + 1)^2 + (x1 - 1 + 1)^2 = x0^2 + x1^2; -B2(R) = -(x0^2 + x1^2)
    assert sp.simplify(expr - (-(x[0]**2 + x[1]**2))) == 0
    Lie = per_mode_lie_derivative(B1, m1)
    # 2 x0 * (-x0) + 2 x1 * (-x1) = -2(x0^2 + x1^2)
    assert sp.simplify(Lie - (-2*(x[0]**2 + x[1]**2))) == 0


def test_hybrid_crossing_count_helpers():
    from src.functions.hybrid import crossing_count_invariant, even_count_equality
    n = sp.Symbol('n')
    polys = crossing_count_invariant(n, k_max=2)
    assert len(polys) == 2
    eq = even_count_equality(n, k_max=1)
    # n * (n - 2) at k_max=1
    assert sp.simplify(eq - n*(n - 2)) == 0


def test_dae_elimination_simple():
    from src.functions.dae import eliminate_algebraic, substitute_algebraic
    x, y = sp.symbols('x y')
    g = [x + y - 1]
    sol = eliminate_algebraic(g, [y], [x])
    assert sol is not None and y in sol
    assert sp.simplify(sol[y] - (1 - x)) == 0
    f_subbed = substitute_algebraic([y * x], sol)
    assert sp.simplify(f_subbed[0] - (1 - x)*x) == 0


def test_slow_fast_quasi_steady():
    from src.functions.slow_fast import quasi_steady_substitution, reduce_dynamics
    xs, xf = sp.symbols('xs xf')
    f_fast = [xs - xf]  # fast eq: xf = xs
    sol = quasi_steady_substitution(f_fast, [xs], [xf])
    assert sp.simplify(sol[xf] - xs) == 0
    f_slow = [xf - 0.1*xs]
    reduced = reduce_dynamics(f_slow, sol)
    assert sp.simplify(reduced[0] - 0.9*xs) == 0


def test_block_decomp_small_gain():
    from src.functions.block_decomp import per_subsystem_lie, supply_rate_polynomial, small_gain_inequality
    x = sp.Symbol('x')
    B = x**2
    f = [-x]
    lie = per_subsystem_lie(B, [x], f)
    assert sp.simplify(lie - (-2*x**2)) == 0
    s = supply_rate_polynomial([x], [x], -x**2)
    total = small_gain_inequality([lie], [s])
    assert sp.simplify(total - (-3*x**2)) == 0


def test_stochastic_subgaussian_lie():
    from src.functions.stochastic_ext import expected_lie_subgaussian
    x = sp.symbols('x0:2')
    B = x[0]**2 + x[1]**2
    f = [-x[0], -x[1]]
    sigma = sp.Matrix([[1, 0], [0, 1]])
    out = expected_lie_subgaussian(B, x, f, sigma, sub_gaussian_proxy_var=1)
    # drift = 2x0*(-x0) + 2x1*(-x1) = -2(x0^2 + x1^2)
    # diff = 0.5 * trace(sigma^T Hess sigma) * 1 = 0.5 * (2 + 2) = 2
    # total = 2 - 2(x0^2 + x1^2)
    assert sp.simplify(out - (2 - 2*(x[0]**2 + x[1]**2))) == 0


def test_nn_per_cell_dynamics():
    from src.functions.nn_control import per_cell_dynamics
    x = sp.symbols('x0:2')
    u = sp.symbols('u0:1')
    f = [x[1], -x[0] + u[0]]
    W = np.array([[1.0, 0.0]])
    b = np.array([0.5])
    closed = per_cell_dynamics(f, x, u, W, b)
    # u = 1*x0 + 0*x1 + 0.5 = x0 + 0.5
    # closed = [x1, -x0 + x0 + 0.5] = [x1, 0.5]
    assert sp.simplify(closed[1] - sp.Rational(1, 2)) == 0


def test_disturbance_wrapper_signature():
    """Verify ct_DS_disturbed exists and accepts the documented signature."""
    from src.functions.disturbance import ct_DS_disturbed
    import inspect
    sig = inspect.signature(ct_DS_disturbed)
    expected = {'b_degree', 'dim', 'L_initial', 'U_initial', 'L_unsafe', 'U_unsafe',
                'L_space', 'U_space', 'x', 'f', 'w_syms', 'W_lo', 'W_hi'}
    assert expected.issubset(sig.parameters.keys())


def test_piecewise_input_substitute_and_consistency():
    from src.functions.piecewise_input import substitute_input, consistency_inclusion_polynomial
    x = sp.symbols('x0:1')
    u = sp.symbols('u0:1')
    f = [x[0] + u[0]]
    fu = substitute_input(f, u, [3])
    assert sp.simplify(fu[0] - (x[0] + 3)) == 0
    head, side = consistency_inclusion_polynomial(B_k=x[0]**2, gamma_k=1,
                                                  B_kplus1=x[0]**2 + 1, gamma_kplus1=2)
    assert sp.simplify(head - (2 - x[0]**2 - 1)) == 0
    assert sp.simplify(side - (1 - x[0]**2)) == 0


def test_v2_imports_clean():
    """All v2 modules must import without error."""
    import src.functions.sets
    import src.functions.pade
    import src.functions.vertex_enumeration
    import src.functions.disturbance
    import src.functions.piecewise_input
    import src.functions.finite_time
    import src.functions.reach_avoid
    import src.functions.hybrid
    import src.functions.dae
    import src.functions.sparse_sos
    import src.functions.slow_fast
    import src.functions.block_decomp
    import src.functions.stochastic_ext
    import src.functions.nn_control


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
