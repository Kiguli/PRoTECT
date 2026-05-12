"""
Continuous-time stochastic barrier with parameter robustness (PRoTECT v2).

Given a stochastic differential equation with optional uncertain parameter:
    dx = f(x, p) dt + delta(x) dW + jumps,    p in [P_lo, P_hi]

find a c-martingale barrier B(x) (independent of p) such that
    sup_{p in P}  A^p B(x) <= c     on the state space X,
where A^p is the infinitesimal generator at parameter p:
    A^p B = <grad B, f(x, p)> + 0.5 * tr(delta^T delta * Hess B)
          + sum_j p_rate_j * (B(x + rho_j) - B(x))    (jump terms)

The safety guarantee (Prajna et al.) then bounds the unsafe-reach
probability over horizon T by gamma + c T <= lambda. Time-bounded
confidence is encoded directly via the level-set constraint
lambda - gamma - c T > 0 (or, with `confidence` parameter, a tighter
threshold).

Differences from v1 ct_SS:
  * uncertain parameter box [P_lo, P_hi] with parameter-box S-procedure
    multipliers Lp(x, p) in the generator constraint
  * margin / mosek_tol / validate_sos surface matching ct_DS_robust
  * full-precision (precision=20) barrier save
  * combined coefficient + pointwise validation
"""

import time as _time
import numpy as np
import sympy as sp

import picos
from SumOfSquares import SOSProblem, poly_variable

from .generate_polynomial import generate_polynomial


def ct_SS_robust(
    b_degree, dim,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    x, f, delta, rho, p_rate, T,
    p_syms=(), P_lo=(), P_hi=(),
    optimize=False, confidence=None,
    solver='mosek',
    gam=None, lam=None, c_val=None, l_degree=None,
    margin=0.0, mosek_tol=None,
    validate_sos=False, validate_tolerance=1e-8,
    init_margin=0.0, unsafe_margin=0.0, generator_margin=0.0,
):
    result = {'b_degree': b_degree}

    n = len(x)
    if not (len(L_initial) == dim == len(U_initial) == len(L_space) == len(U_space) == n == len(f)):
        raise ValueError("length of state arrays doesn't match dimensions!")
    m = len(p_syms)
    if not (len(P_lo) == m == len(P_hi)):
        raise ValueError("p_syms, P_lo, P_hi must agree in length")

    if l_degree is None:
        l_degree = b_degree

    n_unsafe = len(L_unsafe)
    if n_unsafe != len(U_unsafe):
        raise ValueError("Unsafe regions were not defined correctly.")

    g0_polys = generate_polynomial(x, L_initial, U_initial)
    g1_polys = [generate_polynomial(x, L_unsafe[j], U_unsafe[j]) for j in range(n_unsafe)]
    g_space  = generate_polynomial(x, L_space, U_space)
    g_param  = [(p_syms[k] - P_lo[k]) * (P_hi[k] - p_syms[k]) for k in range(m)]
    xp = list(x) + list(p_syms)

    prob = SOSProblem()

    try:
        Barrier = poly_variable('Barrier', x, b_degree)
        L0 = [poly_variable(f'L0_{i+1}', x, l_degree) for i in range(n)]
        L1 = [[poly_variable(f'La_{j}_{i+1}', x, l_degree) for i in range(n)]
              for j in range(n_unsafe)]
        Ls = [poly_variable(f'Ls_{i+1}', xp, l_degree) for i in range(n)]
        Lp = [poly_variable(f'Lp_{k+1}', xp, l_degree) for k in range(m)]

        if gam is None:
            gamma = sp.symbols('gamma_ctss'); gv = prob.sym_to_var(gamma); prob.add_constraint(gv > 0)
        else:
            if gam < 0: raise Exception("Gamma is less than zero!")
            gamma = gam
        if c_val is None:
            c = sp.symbols('c_ctss'); cv = prob.sym_to_var(c); prob.add_constraint(cv > 0)
        else:
            if c_val < 0: raise Exception("c is less than zero!")
            c = c_val
        if lam is None:
            lambda_ = sp.symbols('lambda_ctss'); lv = prob.sym_to_var(lambda_); prob.add_constraint(lv > 0)
        else:
            if lam < 0: raise Exception("Lambda is less than zero!")
            lambda_ = lam

        # Level-set / confidence constraint (Prajna-style time-bounded bound).
        margin_val = float(margin)
        if confidence is None or confidence == 0:
            lam_expr = (lambda_ if lam is not None else lv)
            gam_expr = (gamma if gam is not None else gv)
            c_expr   = (c if c_val is not None else cv)
            sep_expr = lam_expr - gam_expr - c_expr * T
            if margin_val > 0:
                prob.add_constraint(sep_expr >= margin_val)
            else:
                prob.add_constraint(sep_expr > 0)
        else:
            lam_expr = (lambda_ if lam is not None else lv)
            gam_expr = (gamma if gam is not None else gv)
            c_expr   = (c if c_val is not None else cv)
            sep_expr = lam_expr * (1 - confidence) - gam_expr - c_expr * T
            if margin_val > 0:
                prob.add_constraint(sep_expr >= margin_val)
            else:
                prob.add_constraint(sep_expr > 0)
    except Exception:
        return {'error': 'Gamma/Lambda/c definition issues', 'b_degree': b_degree}

    # Infinitesimal generator A^p B(x).
    grad1 = np.array([sp.diff(Barrier, xi) for xi in x])
    grad2 = np.array([[sp.diff(d1, xi) for xi in x] for d1 in grad1])
    jump_term = 0
    for j in range(len(x)):
        Bj = Barrier.subs(x[j], x[j] + rho[j])
        jump_term += p_rate[j] * (Bj - Barrier)
    Barrier_gen = (np.sum(grad1 * f)
                   + 0.5 * np.trace((np.transpose(delta) @ delta) * grad2)
                   + jump_term)

    # Per-condition strict-positivity margins (see ct_DS_robust).
    init_delta = float(init_margin)
    unsafe_delta = float(unsafe_margin)
    gen_delta = float(generator_margin)

    try:
        first_condition = prob.add_sos_constraint(
            -Barrier - sum(Li * gi for Li, gi in zip(L0, g0_polys)) + gamma - init_delta, x)

        for j in range(n_unsafe):
            second_condition = prob.add_sos_constraint(
                Barrier - sum(Lji * gji for Lji, gji in zip(L1[j], g1_polys[j])) - lambda_ - unsafe_delta, x)

        last_condition = prob.add_sos_constraint(
            -Barrier_gen + c
            - sum(Lsi * gi for Lsi, gi in zip(Ls, g_space))
            - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param))
            - gen_delta,
            xp)

        barrier_constraint = prob.add_sos_constraint(Barrier, x)

        for i in L0: prob.add_sos_constraint(i, x)
        for j in range(n_unsafe):
            for i in L1[j]: prob.add_sos_constraint(i, x)
        for i in Ls: prob.add_sos_constraint(i, xp)
        for k in Lp: prob.add_sos_constraint(k, xp)
    except AssertionError:
        return {'error': 'AssertionError (probably odd b_degree)', 'b_degree': b_degree}

    if optimize:
        if confidence is None or confidence == 0:
            prob.set_objective('max',
                (lambda_ if lam is not None else lv)
                - (gamma if gam is not None else gv)
                - (c if c_val is not None else cv) * T)
        else:
            prob.set_objective('max',
                (lambda_ if lam is not None else lv) * (1 - confidence)
                - (gamma if gam is not None else gv)
                - (c if c_val is not None else cv) * T)

    _t0 = _time.time()
    try:
        if solver == 'mosek' and mosek_tol is not None:
            try:
                prob.solve(solver=solver, mosek_params={
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   float(mosek_tol),
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   float(mosek_tol),
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': float(mosek_tol),
                })
            except TypeError:
                prob.solve(solver=solver)
        else:
            prob.solve(solver=solver)
    except picos.modeling.problem.SolutionFailure:
        return {'error': 'picos SolutionFailure', 'b_degree': b_degree,
                'solve_time': _time.time() - _t0}
    except Exception:
        return {'error': 'Solver Exception', 'b_degree': b_degree,
                'solve_time': _time.time() - _t0}
    result['solve_time'] = _time.time() - _t0

    if (len(barrier_constraint.get_sos_decomp()) > 0 and
        len(first_condition.get_sos_decomp()) > 0 and
        len(second_condition.get_sos_decomp()) > 0 and
        len(last_condition.get_sos_decomp()) > 0):
        try:
            result['barrier'] = sum(barrier_constraint.get_sos_decomp(precision=20))
        except Exception:
            result['barrier'] = sum(barrier_constraint.get_sos_decomp())
    else:
        return {'error': 'constraints are not sum of squares'}

    result['gamma']  = float(gv) if gam is None else gam
    result['lambda'] = float(lv) if lam is None else lam
    result['c']      = float(cv) if c_val is None else c_val
    if confidence is None or confidence == 0:
        result['confidence'] = 1 - (result['gamma'] + result['c'] * T) / result['lambda']
    else:
        result['confidence'] = confidence

    if validate_sos:
        from .sos_validate import (validate_problem, pointwise_validate,
                                   pointwise_verdict)
        named = [
            ('init', first_condition,
             -Barrier - sum(Li * gi for Li, gi in zip(L0, g0_polys)) + gamma - init_delta, list(x)),
            ('generator', last_condition,
             -Barrier_gen + c
             - sum(Lsi * gi for Lsi, gi in zip(Ls, g_space))
             - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param))
             - gen_delta,
             list(xp)),
            ('barrier', barrier_constraint, Barrier, list(x)),
            ('unsafe_last', second_condition,
             Barrier - sum(Li * gi for Li, gi in zip(L1[-1], g1_polys[-1])) - lambda_ - unsafe_delta, list(x)),
        ]
        v = validate_problem(prob, named, tolerance=validate_tolerance)
        result['sos_residuals'] = {
            kk: (rv if isinstance(rv, (int, float)) else rv[0])
            for kk, rv in v['residuals'].items()
        }
        result['sos_status']  = v['status']
        result['sos_overall'] = v['overall']

        try:
            B_saved = result.get('barrier', Barrier)
            unsafe_pairs = list(zip(L_unsafe, U_unsafe))
            if len(p_syms):
                import itertools as _it
                p_lo = np.asarray(P_lo, float); p_hi = np.asarray(P_hi, float)
                p_mid = 0.5 * (p_lo + p_hi)
                p_samples = [tuple(p_mid)]
                for bits in _it.product([0, 1], repeat=len(p_syms)):
                    p_samples.append(tuple(
                        p_lo[i] if b == 0 else p_hi[i] for i, b in enumerate(bits)
                    ))
            else:
                p_samples = []
            pw = pointwise_validate(
                B_saved, list(x), result['gamma'], result['lambda'],
                L_initial, U_initial, unsafe_pairs, L_space, U_space,
                dynamics_exprs=None,
                p_syms=list(p_syms), p_samples=p_samples,
            )
            verdict = pointwise_verdict(pw, tolerance=validate_tolerance)
            pw['verdict'] = verdict
            result['pointwise'] = {
                'init_slack':         pw['init_slack'],
                'unsafe_slack':       pw['unsafe_slack'],
                'verdict':            verdict,
            }
            old = result['sos_overall']
            if old == 'clean' and verdict == 'pass':
                result['sos_overall'] = 'clean'
            elif old in ('clean', 'warning') and verdict == 'warn':
                result['sos_overall'] = 'warning'
            else:
                result['sos_overall'] = 'fail'
        except Exception as exc:
            result['pointwise'] = {'error': f'pointwise eval failed: {exc}'}

    return result
