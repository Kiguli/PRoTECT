"""
Continuous-time stochastic-system barrier for SUB-GAUSSIAN noise
(PRoTECT v2 feature 9, end-to-end SOS solver).

For systems
    dx = f(x) dt + sigma(x) dW
where dW is a sub-Gaussian increment with proxy variance kappa
(kappa = 1 recovers standard Gaussian), the worst-case Lie expectation
under the noise's MGF class bounds is

    L_kappa B(x) = grad B . f(x) + 0.5 * kappa * trace(sigma^T Hess(B) sigma)

The barrier conditions become

    B(x) <= gamma     on initial set
    B(x) >  lambda    on unsafe regions
    L_kappa B(x) <= 0 on state space

with lambda > gamma > 0. PRoTECT's existing Gaussian solver (ct_SS) is
the kappa = 1 special case; v2 generalises to arbitrary sub-Gaussian
proxy variance.
"""

import numpy as np
import sympy as sp

import picos
from SumOfSquares import SOSProblem, poly_variable

from .generate_polynomial import generate_polynomial


def ct_SS_subgaussian(
    b_degree,
    x, f, sigma,
    initial_polys=None,
    unsafe_regions=None,
    space_polys=None,
    L_initial=None, U_initial=None,
    L_unsafe=None,  U_unsafe=None,
    L_space=None,   U_space=None,
    sub_gaussian_proxy=1.0,
    solver='mosek',
    gam=None, lam=None, l_degree=None,
):
    n = len(x)
    if len(f) != n:
        raise ValueError("len(f) must equal len(x)")
    sigma = sp.Matrix(sigma)
    if sigma.rows != n:
        raise ValueError("sigma rows must equal len(x)")

    g_init = [sp.sympify(g) for g in initial_polys] if initial_polys else \
             list(generate_polynomial(x, L_initial, U_initial))
    g_space = [sp.sympify(g) for g in space_polys] if space_polys else \
              list(generate_polynomial(x, L_space, U_space))
    g_unsafe = [[sp.sympify(g) for g in r] for r in unsafe_regions] \
        if unsafe_regions else \
        [list(generate_polynomial(x, Lj, Uj)) for Lj, Uj in zip(L_unsafe, U_unsafe)]

    if l_degree is None:
        l_degree = b_degree

    prob = SOSProblem()
    result = {'b_degree': b_degree}

    try:
        Barrier = poly_variable('B_sg', x, b_degree)
        L_i = [poly_variable(f'Lsg_init_{i+1}', x, l_degree)
               for i in range(len(g_init))]
        L_u = [
            [poly_variable(f'Lsg_unsafe_{j}_{i+1}', x, l_degree)
             for i in range(len(g_unsafe[j]))]
            for j in range(len(g_unsafe))
        ]
        L_s = [poly_variable(f'Lsg_space_{i+1}', x, l_degree)
               for i in range(len(g_space))]

        if gam is None:
            gamma = sp.symbols('gamma_sg')
            gv = prob.sym_to_var(gamma); prob.add_constraint(gv > 0)
        else:
            gamma = gam
        if lam is None:
            lambda_ = sp.symbols('lambda_sg')
            lv = prob.sym_to_var(lambda_); prob.add_constraint(lv > 0)
        else:
            lambda_ = lam
        if gam is None and lam is None:
            prob.add_constraint(lv - gv > 0)
    except Exception:
        return {'error': 'init failure', 'b_degree': b_degree}

    # Sub-Gaussian Lie:
    # L_kappa B = grad B . f + 0.5 * kappa * trace(sigma^T Hess B sigma)
    grad = [sp.diff(Barrier, xi) for xi in x]
    drift = sum(grad[i] * f[i] for i in range(n))
    Hess = sp.Matrix([[sp.diff(Barrier, xi, xj) for xj in x] for xi in x])
    diff_term = sp.Rational(1, 2) * sp.sympify(sub_gaussian_proxy) \
        * (sigma.T * Hess * sigma).trace()
    Lie_sg = drift + diff_term

    try:
        first = prob.add_sos_constraint(
            -Barrier
            - sum(Li * gi for Li, gi in zip(L_i, g_init))
            + gamma,
            x,
        )
        last_unsafe = None
        for j in range(len(g_unsafe)):
            last_unsafe = prob.add_sos_constraint(
                Barrier
                - sum(Li * gi for Li, gi in zip(L_u[j], g_unsafe[j]))
                - lambda_,
                x,
            )
        lie_cond = prob.add_sos_constraint(
            -Lie_sg
            - sum(Li * gi for Li, gi in zip(L_s, g_space)),
            x,
        )

        for Li in L_i:
            prob.add_sos_constraint(Li, x)
        for region in L_u:
            for Li in region:
                prob.add_sos_constraint(Li, x)
        for Li in L_s:
            prob.add_sos_constraint(Li, x)

        b_cond = prob.add_sos_constraint(Barrier, x)

    except AssertionError:
        return {'error': 'AssertionError', 'b_degree': b_degree}

    try:
        prob.solve(solver=solver)
    except picos.modeling.problem.SolutionFailure:
        return {'error': 'picos SolutionFailure', 'b_degree': b_degree}
    except Exception:
        return {'error': 'Solver Exception', 'b_degree': b_degree}

    try:
        if (len(b_cond.get_sos_decomp()) > 0
            and len(first.get_sos_decomp()) > 0
            and last_unsafe is not None
            and len(last_unsafe.get_sos_decomp()) > 0
            and len(lie_cond.get_sos_decomp()) > 0):
            result['barrier'] = sum(b_cond.get_sos_decomp())
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
        return {'error': 'level-set numerical issue',
                'b_degree': b_degree}
    except Exception:
        return {'error': 'reading result failed', 'b_degree': b_degree}
