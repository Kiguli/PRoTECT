"""
ct_DS_v2 -- unified continuous-time deterministic-system barrier solver
for PRoTECT v2.

Extends ct_DS with:
    * arbitrary polynomial-inequality sets for initial / unsafe / state
      space (drop-in for axis-aligned boxes; integrates v2 features
      2a / 2b / 2c via sets.py),
    * uncertain parameters / disturbances / aux variables with box
      bounds via parameter-box S-procedure multipliers (features 4a,
      7a; treats sinc / Pade / sqrt / inv_power auxiliary states from
      relaxations.py uniformly),
    * polynomial-equality multipliers (features 1a-sqrt, 1a-inv_power,
      1b Pade, 3b even-count parity, 5b manifold-restricted SOS).

The standard time-unbounded safety condition is preserved; for finite-
time reach-tube width use ct_DS_finite_time, for reach-avoid use
ct_DS_reach_avoid.

API (positional args mirror ct_DS where possible):
    b_degree         degree of the barrier polynomial in x.
    x                list of sympy state symbols, length n.
    f                list/array of sympy expressions (the dynamics);
                     may depend on (x, p, aux).
    initial_polys    list of sympy polys, each >= 0 inside the initial
                     set. (Use sets.box_to_polytope(x, L, U) to get the
                     legacy axis-aligned-box form.)
    unsafe_regions   list of lists of sympy polys; entry j is the
                     polynomial-inequality conjunction for unsafe
                     region j.
    space_polys      list of sympy polys, each >= 0 inside the state
                     space (the Lie SOS multiplier g list).
    equality_polys   optional list of polys h_k; each enforced via a
                     FREE polynomial Lagrangian lambda_k * h_k added to
                     the Lie SOS expression.
    p_syms, P_lo, P_hi
                     optional uncertain parameter / disturbance / aux
                     variable symbols and their box ranges. Treated as
                     box S-procedure multipliers in the Lie SOS.
    solver, gam, lam, l_degree
                     same as ct_DS.

Returns the same dict shape as ct_DS:
    {'b_degree', 'barrier', 'gamma', 'lambda'}    on success
    {'error', 'b_degree'}                          on failure
"""

import numpy as np
import sympy as sp

import picos
from SumOfSquares import SOSProblem, poly_variable

from .generate_polynomial import generate_polynomial
from .sets import box_to_polytope


def _resolve_set(x, polys, L=None, U=None):
    """Accept polynomial-inequality list directly OR fall back to a box
    description (using the canonical (x_i-L_i)*(U_i-x_i) quadratic form
    that matches the existing ct_DS pipeline). At least one input must
    be provided."""
    if polys is not None:
        return [sp.sympify(g) for g in polys]
    if L is None or U is None:
        raise ValueError(
            "must provide either polys or both L and U for the set")
    return list(generate_polynomial(x, L, U))


def _resolve_unsafe(x, regions_polys, L_unsafe=None, U_unsafe=None):
    """For unsafe regions: list of inequality lists (one per region) or
    box arrays."""
    if regions_polys is not None:
        return [[sp.sympify(g) for g in region] for region in regions_polys]
    if L_unsafe is None or U_unsafe is None:
        raise ValueError(
            "must provide either unsafe_regions or both L_unsafe and U_unsafe")
    return [list(generate_polynomial(x, Lj, Uj))
            for Lj, Uj in zip(L_unsafe, U_unsafe)]


def ct_DS_v2(
    b_degree,
    x, f,
    initial_polys=None,
    unsafe_regions=None,
    space_polys=None,
    L_initial=None, U_initial=None,
    L_unsafe=None,  U_unsafe=None,
    L_space=None,   U_space=None,
    equality_polys=(),
    p_syms=(), P_lo=(), P_hi=(),
    solver='mosek',
    gam=None, lam=None, l_degree=None,
):
    n = len(x)
    if len(f) != n:
        raise ValueError("len(f) must equal len(x)")

    g0_polys = _resolve_set(x, initial_polys, L_initial, U_initial)
    g_space = _resolve_set(x, space_polys, L_space, U_space)
    g_unsafe = _resolve_unsafe(x, unsafe_regions, L_unsafe, U_unsafe)

    p_syms = list(p_syms)
    m = len(p_syms)
    if not (len(P_lo) == m == len(P_hi)):
        raise ValueError("p_syms / P_lo / P_hi length mismatch")
    g_param = [(p_syms[k] - P_lo[k]) * (P_hi[k] - p_syms[k]) for k in range(m)]

    equality_polys = [sp.sympify(h) for h in equality_polys]

    xp = list(x) + p_syms

    if l_degree is None:
        l_degree = b_degree

    prob = SOSProblem()
    result = {'b_degree': b_degree}

    try:
        Barrier = poly_variable('Barrier', x, b_degree)

        L0 = [poly_variable(f'L0_{i+1}', x, l_degree)
              for i in range(len(g0_polys))]
        L1 = [
            [poly_variable(f'La_{j}_{i+1}', x, l_degree)
             for i in range(len(g_unsafe[j]))]
            for j in range(len(g_unsafe))
        ]
        Ls = [poly_variable(f'Ls_{i+1}', xp, l_degree)
              for i in range(len(g_space))]
        Lp = [poly_variable(f'Lp_{k+1}', xp, l_degree)
              for k in range(m)]
        # Equality multipliers: free polynomials in (x, p) -- NO SOS test.
        Le = [poly_variable(f'Le_{k+1}', xp, l_degree)
              for k in range(len(equality_polys))]

        if gam is None:
            gamma = sp.symbols('gamma_v2')
            gv = prob.sym_to_var(gamma)
            prob.add_constraint(gv > 0)
        else:
            if gam < 0:
                raise Exception("Gamma is less than zero!")
            gamma = gam

        if lam is None:
            lambda_ = sp.symbols('lambda_v2')
            lv = prob.sym_to_var(lambda_)
            prob.add_constraint(lv > 0)
        else:
            if lam < 0:
                raise Exception("Lambda is less than zero!")
            lambda_ = lam

        if gam is None and lam is None:
            prob.add_constraint(lv - gv > 0)
        elif gam is None:
            prob.add_constraint(lambda_ - gv > 0)
        elif lam is None:
            prob.add_constraint(lv - gamma > 0)
        else:
            if lam <= gam:
                raise Exception(
                    "User defined lambda value is less than user defined gamma!"
                )
    except Exception:
        return {'error': 'Gamma or Lambda definition issues',
                'b_degree': b_degree}

    LieDeriv = np.array([sp.diff(Barrier, xi) for xi in x])
    Barrier_f = np.sum(LieDeriv * np.array(f))

    try:
        first_condition = prob.add_sos_constraint(
            -Barrier - sum(Li * gi for Li, gi in zip(L0, g0_polys)) + gamma,
            x,
        )

        last_unsafe_condition = None
        for j in range(len(g_unsafe)):
            cond_j = prob.add_sos_constraint(
                Barrier - sum(Li * gi for Li, gi in zip(L1[j], g_unsafe[j])) - lambda_,
                x,
            )
            last_unsafe_condition = cond_j

        Lie_terms = -Barrier_f
        Lie_terms = Lie_terms - sum(
            Li * gi for Li, gi in zip(Ls, g_space))
        Lie_terms = Lie_terms - sum(
            Lk * gk for Lk, gk in zip(Lp, g_param))
        Lie_terms = Lie_terms - sum(
            Lh * h for Lh, h in zip(Le, equality_polys))
        last_condition = prob.add_sos_constraint(Lie_terms, xp)

        barrier_constraint = prob.add_sos_constraint(Barrier, x)

        for Li in L0:
            prob.add_sos_constraint(Li, x)
        for Lj in L1:
            for Li in Lj:
                prob.add_sos_constraint(Li, x)
        for Li in Ls:
            prob.add_sos_constraint(Li, xp)
        for Lk in Lp:
            prob.add_sos_constraint(Lk, xp)
        # Le entries are free, no SOS test (equality multipliers).
    except AssertionError:
        return {'error': 'AssertionError (probably odd b_degree)',
                'b_degree': b_degree}

    try:
        prob.solve(solver=solver)
    except picos.modeling.problem.SolutionFailure:
        return {'error': 'picos SolutionFailure', 'b_degree': b_degree}
    except Exception:
        return {'error': 'Solver Exception', 'b_degree': b_degree}

    if len(barrier_constraint.get_sos_decomp().free_symbols) == 0:
        return {'error': 'barrier is scalar!', 'b_degree': b_degree}

    if (len(barrier_constraint.get_sos_decomp()) > 0 and
        len(first_condition.get_sos_decomp()) > 0 and
        last_unsafe_condition is not None and
        len(last_unsafe_condition.get_sos_decomp()) > 0 and
        len(last_condition.get_sos_decomp()) > 0):
        result['barrier'] = sum(barrier_constraint.get_sos_decomp())
    else:
        return {'error': 'constraints are not sum of squares'}

    if gam is None:
        result['gamma'] = float(gv)
    else:
        result['gamma'] = gam
    if lam is None:
        result['lambda'] = float(lv)
    else:
        result['lambda'] = lam

    if result['lambda'] > result['gamma'] and result['lambda'] > 0 and result['gamma'] > 0:
        return result
    elif result['lambda'] <= result['gamma']:
        return {'error': 'lambda not greater than gamma',
                'b_degree': b_degree}
    return {'error': 'numerical error on level sets e.g. negative value',
            'b_degree': b_degree}
