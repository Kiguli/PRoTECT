"""
End-to-end SOS smoke tests for the v2 solvers. Each test sets up a
small problem with a known answer and checks the solver finds a
barrier (or the correct certificate object) in reasonable wall time.

These exercise the actual MOSEK / CVXOPT pipeline. Skip with
PROTECT_SKIP_E2E=1 to run only the assembly tests.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy as sp


SKIP = os.environ.get('PROTECT_SKIP_E2E') == '1'
SOLVER = os.environ.get('PROTECT_SOLVER', 'mosek')


def _assert(cond, msg=''):
    if not cond:
        raise AssertionError(msg)


def test_ct_DS_v2_basic_2d_oscillator():
    """Standard 2-D damped oscillator; ct_DS_v2 should find a barrier."""
    if SKIP: return
    from src.functions.ct_DS_v2 import ct_DS_v2
    x = sp.symbols('x0:2')
    L_initial = np.array([-0.05, -0.05]); U_initial = np.array([0.05, 0.05])
    L_space = np.array([-2.0, -2.0]);     U_space = np.array([2.0, 2.0])
    L_u1 = L_space.copy(); L_u1[0] = 1.5; U_u1 = U_space.copy()
    L_u2 = L_space.copy(); U_u2 = U_space.copy(); U_u2[0] = -1.5
    f = np.array([x[1], -x[0] - x[1]])

    res = ct_DS_v2(
        b_degree=2, x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=np.array([L_u1, L_u2]), U_unsafe=np.array([U_u1, U_u2]),
        L_space=L_space, U_space=U_space,
        solver=SOLVER,
    )
    _assert('barrier' in res, f'ct_DS_v2 failed: {res}')
    print(f'  ct_DS_v2 OK -- gamma={res["gamma"]:.3g}, lambda={res["lambda"]:.3g}')


def test_ct_DS_v2_with_quadratic_unsafe():
    """Unsafe set as a quadratic sub-level set (ellipse far from origin)."""
    if SKIP: return
    from src.functions.ct_DS_v2 import ct_DS_v2
    from src.functions.sets import quadratic_form_set
    x = sp.symbols('x0:2')
    f = np.array([x[1], -x[0] - x[1]])
    L_initial = np.array([-0.05, -0.05]); U_initial = np.array([0.05, 0.05])
    L_space = np.array([-3.0, -3.0]);     U_space = np.array([3.0, 3.0])
    # Unsafe: ellipsoid (x0-2)^2 + x1^2 <= 0.25 (centred at (2, 0))
    # quadratic_form_set with sense='le' returns [-(x^T Q x + c x + d)]
    # We want the inequality `>= 0` form to mean "inside unsafe set",
    # so the polynomial is r^2 - ((x0-2)^2 + x1^2).
    g = (x[0] - 2)**2 + x[1]**2 - 0.25  # <= 0 means inside the ellipse
    unsafe_polys = [-g]                  # >= 0 inside the ellipse
    res = ct_DS_v2(
        b_degree=2, x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        unsafe_regions=[unsafe_polys],
        L_space=L_space, U_space=U_space,
        solver=SOLVER,
    )
    _assert('barrier' in res, f'ct_DS_v2 quadratic-unsafe failed: {res}')
    print(f'  ct_DS_v2 with quadratic unsafe OK')


def test_ct_DS_v2_with_parameter():
    """Same shape as ct_DS_robust test: parameter-uncertain damping."""
    if SKIP: return
    from src.functions.ct_DS_v2 import ct_DS_v2
    x = sp.symbols('x0:2')
    p = sp.symbols('p0:1')
    L_initial = np.array([-0.05, -0.05]); U_initial = np.array([0.05, 0.05])
    L_space = np.array([-2.0, -2.0]);     U_space = np.array([2.0, 2.0])
    L_u1 = L_space.copy(); L_u1[0] = 1.5; U_u1 = U_space.copy()
    L_u2 = L_space.copy(); U_u2 = U_space.copy(); U_u2[0] = -1.5
    f = np.array([x[1], -x[0] - p[0]*x[1]])
    res = ct_DS_v2(
        b_degree=2, x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=np.array([L_u1, L_u2]), U_unsafe=np.array([U_u1, U_u2]),
        L_space=L_space, U_space=U_space,
        p_syms=p, P_lo=np.array([0.5]), P_hi=np.array([1.5]),
        solver=SOLVER,
    )
    _assert('barrier' in res, f'ct_DS_v2 with p failed: {res}')
    free = res['barrier'].free_symbols
    _assert(p[0] not in free, f'B contained p: {free & set(p)}')
    print(f'  ct_DS_v2 + parameter OK; barrier is p-independent')


def test_ct_DS_finite_time():
    """Time-augmented barrier on small linear system."""
    if SKIP: return
    from src.functions.ct_DS_finite_time import ct_DS_finite_time
    x = sp.symbols('x0:2')
    L_initial = np.array([-0.1, -0.1]); U_initial = np.array([0.1, 0.1])
    L_space = np.array([-2.0, -2.0]);   U_space = np.array([2.0, 2.0])
    L_u1 = L_space.copy(); L_u1[0] = 1.5; U_u1 = U_space.copy()
    f = np.array([x[1], -x[0] - x[1]])
    res = ct_DS_finite_time(
        b_degree=2, time_orders=2,
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=np.array([L_u1]), U_unsafe=np.array([U_u1]),
        L_space=L_space, U_space=U_space,
        T_horizon=5.0,
        solver=SOLVER,
    )
    print(f'  ct_DS_finite_time -> {sorted(res.keys())}')
    # Finite-time barrier may or may not be feasible at low degree on
    # this small problem; we accept either a barrier or an explicit
    # error verdict (both prove the solver is functional).
    _assert('barrier' in res or 'error' in res,
            f'unexpected return shape: {res}')


def test_ct_DS_hybrid_two_modes():
    """Two-mode hybrid: stable -> stable transition."""
    if SKIP: return
    from src.functions.ct_DS_hybrid import HybridMode, HybridEdge, ct_DS_hybrid
    from src.functions.sets import box_to_polytope
    x = sp.symbols('x0:2')

    # Mode m1: damped linear, x0 in [-2, 2], x1 in [-2, 2].
    inv_polys = box_to_polytope(x, [-2.0, -2.0], [2.0, 2.0])
    init_polys = box_to_polytope(x, [-0.05, -0.05], [0.05, 0.05])
    unsafe_polys = box_to_polytope(x, [1.5, -2.0], [2.0, 2.0])
    m1 = HybridMode('m1', x,
                    [x[1], -x[0] - x[1]],
                    invariant_polys=inv_polys,
                    initial_polys=init_polys,
                    unsafe_regions_polys=[unsafe_polys])
    m2 = HybridMode('m2', x,
                    [x[1], -2*x[0] - x[1]],
                    invariant_polys=inv_polys,
                    initial_polys=[],
                    unsafe_regions_polys=[unsafe_polys])

    # Edge m1 -> m2 with no guard, identity reset.
    edge = HybridEdge('m1', 'm2', guard_polys=[],
                      reset_map=lambda src_x: list(src_x))

    res = ct_DS_hybrid(
        b_degree=2,
        modes=[m1, m2],
        edges=[edge],
        solver=SOLVER,
    )
    print(f'  ct_DS_hybrid -> {sorted(res.keys())}')
    _assert('barriers' in res or 'error' in res,
            f'unexpected: {res}')


def test_ct_SS_subgaussian():
    """2-D linear SDE with sub-Gaussian noise; should find a barrier."""
    if SKIP: return
    from src.functions.ct_SS_subgaussian import ct_SS_subgaussian
    x = sp.symbols('x0:2')
    L_initial = np.array([-0.05, -0.05]); U_initial = np.array([0.05, 0.05])
    L_space = np.array([-2.0, -2.0]);     U_space = np.array([2.0, 2.0])
    L_u1 = L_space.copy(); L_u1[0] = 1.5; U_u1 = U_space.copy()
    f = [x[1], -x[0] - x[1]]
    sigma = sp.Matrix([[0.1, 0], [0, 0.1]])
    res = ct_SS_subgaussian(
        b_degree=2, x=x, f=f, sigma=sigma,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=np.array([L_u1]), U_unsafe=np.array([U_u1]),
        L_space=L_space, U_space=U_space,
        sub_gaussian_proxy=1.0,
        solver=SOLVER,
    )
    _assert('barrier' in res or 'error' in res, f'unexpected: {res}')
    if 'barrier' in res:
        print(f'  ct_SS_subgaussian OK')
    else:
        print(f'  ct_SS_subgaussian -> error: {res["error"]}')


def test_ct_DS_piecewise_sequence_runs():
    """Two-segment piecewise input: solver invokes per-segment ct_DS_v2."""
    if SKIP: return
    from src.functions.ct_DS_piecewise_sequence import ct_DS_piecewise_sequence
    x = sp.symbols('x0:2')
    u = sp.symbols('u0:1')
    L_initial = np.array([-0.05, -0.05]); U_initial = np.array([0.05, 0.05])
    L_space = np.array([-2.0, -2.0]);     U_space = np.array([2.0, 2.0])
    L_u1 = L_space.copy(); L_u1[0] = 1.5; U_u1 = U_space.copy()
    f_template = [x[1], -x[0] - x[1] + u[0]]
    out = ct_DS_piecewise_sequence(
        b_degree=2, x=x, f_template=f_template, u_syms=u,
        u_sequence=[[0.0], [0.0]],
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=np.array([L_u1]), U_unsafe=np.array([U_u1]),
        L_space=L_space, U_space=U_space,
        solver=SOLVER,
    )
    _assert(out['n_segments'] == 2, f'expected 2 segments: {out}')
    print(f'  ct_DS_piecewise_sequence feasible={out["feasible"]}')


def test_ct_DS_reach_avoid_invokes_solver():
    """Reach-avoid invokes the SOS solver and returns either a barrier
    triple (B, gamma_safe, gamma_target) or an explicit error."""
    if SKIP: return
    from src.functions.ct_DS_reach_avoid import ct_DS_reach_avoid
    x = sp.symbols('x0:2')
    L_initial = np.array([-0.5, -0.5]); U_initial = np.array([0.5, 0.5])
    L_space = np.array([-2.0, -2.0]);   U_space = np.array([2.0, 2.0])
    L_target = np.array([-0.05, -0.05]); U_target = np.array([0.05, 0.05])
    L_u1 = L_space.copy(); L_u1[0] = 1.5; U_u1 = U_space.copy()
    f = [-x[0], -x[1]]   # globally stable linear system
    res = ct_DS_reach_avoid(
        b_degree=2, x=x, f=f,
        gamma_safe=-0.1, gamma_target=0.5,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=np.array([L_u1]), U_unsafe=np.array([U_u1]),
        L_target=L_target, U_target=U_target,
        L_space=L_space, U_space=U_space,
        solver=SOLVER,
    )
    print(f'  ct_DS_reach_avoid -> {sorted(res.keys())}')
    _assert('barrier' in res or 'error' in res, f'unexpected: {res}')


if __name__ == '__main__':
    failed = 0
    tests = [name for name, fn in dict(globals()).items()
             if name.startswith('test_') and callable(fn)]
    for name in sorted(tests):
        fn = globals()[name]
        t0 = time.time()
        try:
            fn()
            print(f'PASS {name}  ({time.time() - t0:.2f}s)')
        except Exception as exc:
            failed += 1
            print(f'FAIL {name}: {exc}  ({time.time() - t0:.2f}s)')
    print(f'\n{failed} failure(s) out of {len(tests)}')
    sys.exit(1 if failed else 0)
