"""
Discrete-time stochastic barrier with parameter robustness (PRoTECT v2).

Given a discrete-time stochastic system with optional uncertain parameter:
    x_{k+1} = f(x_k, p) + varsigma_k,    p in [P_lo, P_hi],
where varsigma is i.i.d. noise (Gaussian / lognormal / uniform / ...),
find a c-supermartingale barrier B(x) (independent of p) such that
    sup_{p in P}  E[B(x_{k+1}) | x_k]  <=  B(x_k) + c    on X x P.

The unsafe-reach probability over N steps is then bounded by
  gamma + c N <= lambda    =>    P(reach unsafe within N) <= gamma / lambda
                                                              + c N / lambda

This v2 variant adds the parameter-box S-procedure to the expectation
SOS constraint plus the v2 validation surface.

Supported noise_type values: 'normal' / 'gaussian', 'lognormal',
'uniform', 'exponential', 'rayleigh'. Implementation reuses the
moment-substitution machinery from v1 dt_SS (`dt_SS`).
"""

import time as _time
import numpy as np
import sympy as sp
import scipy.stats

import picos
from SumOfSquares import SOSProblem, poly_variable

from .generate_polynomial import generate_polynomial
from .doublefactorial import doublefactorial
from .factorial import factorial


def _substitute_noise_moments(BB, varsigma, noise_type,
                              mean=None, sigma=None, rate=None,
                              a=None, b=None):
    """Substitute the i.i.d. noise expectation into a polynomial BB."""
    BB = sp.expand(BB)
    for i in range(len(varsigma)):
        coeff_dict = BB.as_coefficients_dict()
        m_list = list(coeff_dict.keys())
        coeffs = list(coeff_dict.values())
        new_terms = []
        for k_idx, monom in enumerate(m_list):
            s = str(monom).replace('**', '^').split('*')
            for k2, factor in enumerate(s):
                parts = factor.split('^')
                if parts[0] == varsigma[i].name:
                    p_deg = int(parts[1]) if len(parts) > 1 else 1
                    if noise_type in ('normal', 'gaussian'):
                        df = scipy.stats.norm.moment(p_deg, mean[i], sigma[i])
                    elif noise_type == 'lognormal':
                        df = scipy.stats.lognorm.moment(p_deg, sigma[i], scale=np.exp(mean[i]))
                    elif noise_type == 'uniform':
                        df = scipy.stats.uniform.moment(p_deg, loc=a[i], scale=b[i] - a[i])
                    elif noise_type == 'exponential':
                        df = scipy.stats.expon.moment(p_deg, scale=1.0 / rate[i])
                    elif noise_type == 'rayleigh':
                        df = scipy.stats.rayleigh.moment(p_deg, scale=sigma[i])
                    else:
                        df = 0.0
                    s[k2] = str(df)
            new_terms.append(coeffs[k_idx] * sp.sympify('*'.join(s)))
        BB = sp.expand(sum(new_terms, sp.Integer(0)))
    return BB


def dt_SS_robust(
    b_degree, dim,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    x, varsigma, f, t,
    noise_type='normal',
    p_syms=(), P_lo=(), P_hi=(),
    optimize=False, confidence=None,
    solver='mosek',
    gam=None, lam=None, c_val=None, l_degree=None,
    mean=None, sigma=None, rate=None, a=None, b=None,
    margin=0.0, mosek_tol=None,
    validate_sos=False, validate_tolerance=1e-8,
    init_margin=0.0, unsafe_margin=0.0, expectation_margin=0.0,
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
            gamma = sp.symbols('gamma_dtss'); gv = prob.sym_to_var(gamma); prob.add_constraint(gv > 0)
        else:
            gamma = gam
        if c_val is None:
            c = sp.symbols('c_dtss'); cv = prob.sym_to_var(c); prob.add_constraint(cv > 0)
        else:
            c = c_val
        if lam is None:
            lambda_ = sp.symbols('lambda_dtss'); lv = prob.sym_to_var(lambda_); prob.add_constraint(lv > 0)
        else:
            lambda_ = lam

        margin_val = float(margin)
        lam_expr = (lambda_ if lam is not None else lv)
        gam_expr = (gamma if gam is not None else gv)
        c_expr   = (c if c_val is not None else cv)
        if confidence is None or confidence == 0:
            sep = lam_expr - gam_expr - c_expr * t
        else:
            sep = lam_expr * (1 - confidence) - gam_expr - c_expr * t
        if margin_val > 0:
            prob.add_constraint(sep >= margin_val)
        else:
            prob.add_constraint(sep > 0)
    except Exception:
        return {'error': 'Gamma/Lambda/c definition issues', 'b_degree': b_degree}

    # Build E[B(x_{k+1}) | x_k, p] by substituting f(x, p) + varsigma into B
    # and then substituting noise moments.
    y = [sp.Dummy(f'y{i}') for i in range(n)]
    BB = Barrier.subs([(x[i], y[i]) for i in range(n)])
    BB = BB.subs([(y[i], f[i] + varsigma[i]) for i in range(n)])
    BB = _substitute_noise_moments(BB, varsigma, noise_type,
                                   mean=mean, sigma=sigma, rate=rate, a=a, b=b)

    # Per-condition strict-positivity margins (see ct_DS_robust).
    init_delta = float(init_margin)
    unsafe_delta = float(unsafe_margin)
    exp_delta = float(expectation_margin)

    try:
        first_condition = prob.add_sos_constraint(
            -Barrier - sum(Li * gi for Li, gi in zip(L0, g0_polys)) + gamma - init_delta, x)
        for j in range(n_unsafe):
            second_condition = prob.add_sos_constraint(
                Barrier - sum(Lji * gji for Lji, gji in zip(L1[j], g1_polys[j])) - lambda_ - unsafe_delta, x)
        last_condition = prob.add_sos_constraint(
            -BB + Barrier + c
            - sum(Lsi * gi for Lsi, gi in zip(Ls, g_space))
            - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param))
            - exp_delta,
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
        prob.set_objective('max', sep)

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
        result['confidence'] = 1 - (result['gamma'] + result['c'] * t) / result['lambda']
    else:
        result['confidence'] = confidence

    if validate_sos:
        from .sos_validate import (validate_problem, pointwise_validate,
                                   pointwise_verdict)
        named = [
            ('init', first_condition,
             -Barrier - sum(Li * gi for Li, gi in zip(L0, g0_polys)) + gamma - init_delta, list(x)),
            ('expectation', last_condition,
             -BB + Barrier + c
             - sum(Lsi * gi for Lsi, gi in zip(Ls, g_space))
             - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param))
             - exp_delta,
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
                        p_lo[i] if bv == 0 else p_hi[i] for i, bv in enumerate(bits)
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
                'init_slack':   pw['init_slack'],
                'unsafe_slack': pw['unsafe_slack'],
                'verdict':      verdict,
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
