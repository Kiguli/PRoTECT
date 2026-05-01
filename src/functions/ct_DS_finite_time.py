"""
Finite-time barrier search (PRoTECT v2 feature 6a, end-to-end SOS solver).

Time-augmented barrier ``B(x, t) = sum_k t^k * B_k(x)`` certifying that
trajectories starting in the initial set never enter any unsafe region
during the bounded horizon ``t in [0, T]``. The polynomial Lie
derivative ``dB/dt + <dB/dx, f>`` must be non-positive on the
state-space x time interval rectangle.

SOS conditions:
    B(x, 0)              <= 0          on initial set
    B(x, t)              >  0          on unsafe regions for t in [0, T]
    -(dB/dt + <dB/dx, f>)>= 0          on state x [0, T]

with the time-box S-procedure multiplier g_t(t) = t * (T - t).

This solver is the natural target for ROBE25's paper specification
("width(x+y+z) at t=40s must be < 1e-5"): pick the unsafe region as
``{(x, y, z) : |x + y + z - 1| > 1e-5}``, set T=40, and the solver
either returns a barrier B(x, t) or reports infeasibility.
"""

import numpy as np
import sympy as sp

import picos
from SumOfSquares import SOSProblem, poly_variable

from .generate_polynomial import generate_polynomial


def ct_DS_finite_time(
    b_degree, time_orders,
    x, f,
    initial_polys=None,
    unsafe_regions=None,
    space_polys=None,
    L_initial=None, U_initial=None,
    L_unsafe=None,  U_unsafe=None,
    L_space=None,   U_space=None,
    T_horizon=1.0,
    time_sym=None,
    solver='mosek',
    gam=None, lam=None, l_degree=None,
):
    n = len(x)
    if len(f) != n:
        raise ValueError("len(f) must equal len(x)")

    if time_sym is None:
        time_sym = sp.Symbol('t_finite')
    t = time_sym
    xt = list(x) + [t]

    if initial_polys is not None:
        g_initial = [sp.sympify(g) for g in initial_polys]
    else:
        g_initial = list(generate_polynomial(x, L_initial, U_initial))

    if space_polys is not None:
        g_space = [sp.sympify(g) for g in space_polys]
    else:
        g_space = list(generate_polynomial(x, L_space, U_space))

    if unsafe_regions is not None:
        g_unsafe = [[sp.sympify(g) for g in r] for r in unsafe_regions]
    else:
        g_unsafe = [list(generate_polynomial(x, Lj, Uj))
                    for Lj, Uj in zip(L_unsafe, U_unsafe)]

    g_time = t * (sp.sympify(T_horizon) - t)

    if l_degree is None:
        l_degree = b_degree

    prob = SOSProblem()
    result = {'b_degree': b_degree, 'time_orders': time_orders}

    try:
        # Time-varying barrier.
        Bk_polys = [poly_variable(f'Bk_{k}', x, b_degree)
                    for k in range(time_orders + 1)]
        Barrier = sum(Bk_polys[k] * t**k for k in range(time_orders + 1))

        L_init = [poly_variable(f'L0i_{i+1}', x, l_degree)
                  for i in range(len(g_initial))]
        L_unsafe_per_region = [
            [poly_variable(f'Lu_{j}_{i+1}', xt, l_degree)
             for i in range(len(g_unsafe[j]))]
            for j in range(len(g_unsafe))
        ]
        L_t_unsafe = [poly_variable(f'Lt_unsafe_{j+1}', xt, l_degree)
                      for j in range(len(g_unsafe))]

        Ls = [poly_variable(f'Ls_{i+1}', xt, l_degree)
              for i in range(len(g_space))]
        Lt_lie = poly_variable('Lt_lie', xt, l_degree)

        if gam is None:
            gamma = sp.symbols('gamma_ft')
            gv = prob.sym_to_var(gamma); prob.add_constraint(gv > 0)
        else:
            gamma = gam
        if lam is None:
            lambda_ = sp.symbols('lambda_ft')
            lv = prob.sym_to_var(lambda_); prob.add_constraint(lv > 0)
        else:
            lambda_ = lam
        if gam is None and lam is None:
            prob.add_constraint(lv - gv > 0)
    except Exception:
        return {'error': 'init failure', 'b_degree': b_degree}

    # Lie derivative: dB/dt + grad_x B . f
    dBdt = sp.diff(Barrier, t)
    grad_x = np.array([sp.diff(Barrier, xi) for xi in x])
    Lie = dBdt + np.sum(grad_x * np.array(f))

    Barrier_at_0 = Barrier.subs(t, 0)

    try:
        # Initial: -B(x, 0) - sum L_init_i * g_init_i + gamma SOS in x.
        first_condition = prob.add_sos_constraint(
            -Barrier_at_0
            - sum(Li * gi for Li, gi in zip(L_init, g_initial))
            + gamma,
            x,
        )

        # Unsafe (per region j): B(x, t) - sum L_u * g_unsafe_j - L_t * g_t - lambda SOS in (x, t).
        last_unsafe = None
        for j in range(len(g_unsafe)):
            terms = Barrier
            terms = terms - sum(
                Li * gi for Li, gi in zip(L_unsafe_per_region[j], g_unsafe[j]))
            terms = terms - L_t_unsafe[j] * g_time
            terms = terms - lambda_
            cond_j = prob.add_sos_constraint(terms, xt)
            last_unsafe = cond_j

        # Lie: -Lie - sum Ls_i * g_space_i - Lt_lie * g_t SOS in (x, t).
        last_condition = prob.add_sos_constraint(
            -Lie
            - sum(Li * gi for Li, gi in zip(Ls, g_space))
            - Lt_lie * g_time,
            xt,
        )

        # All multipliers SOS over their respective variable lists.
        for Li in L_init:
            prob.add_sos_constraint(Li, x)
        for region in L_unsafe_per_region:
            for Li in region:
                prob.add_sos_constraint(Li, xt)
        for Lt in L_t_unsafe:
            prob.add_sos_constraint(Lt, xt)
        for Li in Ls:
            prob.add_sos_constraint(Li, xt)
        prob.add_sos_constraint(Lt_lie, xt)

        barrier_constraint = prob.add_sos_constraint(Barrier, xt)

    except AssertionError:
        return {'error': 'AssertionError', 'b_degree': b_degree}

    try:
        prob.solve(solver=solver)
    except picos.modeling.problem.SolutionFailure:
        return {'error': 'picos SolutionFailure', 'b_degree': b_degree}
    except Exception:
        return {'error': 'Solver Exception', 'b_degree': b_degree}

    if (len(barrier_constraint.get_sos_decomp()) > 0
        and len(first_condition.get_sos_decomp()) > 0
        and last_unsafe is not None
        and len(last_unsafe.get_sos_decomp()) > 0
        and len(last_condition.get_sos_decomp()) > 0):
        result['barrier'] = sum(barrier_constraint.get_sos_decomp())
    else:
        return {'error': 'constraints not SOS', 'b_degree': b_degree}

    if gam is None:
        result['gamma'] = float(gv)
    else:
        result['gamma'] = gam
    if lam is None:
        result['lambda'] = float(lv)
    else:
        result['lambda'] = lam

    if result['lambda'] > result['gamma'] > 0:
        return result
    return {'error': 'level-set numerical issue', 'b_degree': b_degree}
