"""
Discrete-time deterministic-system barrier search with PARAMETER ROBUSTNESS
and the v2 validator surface (PRoTECT v2 analogue of ct_DS_robust).

Given discrete-time dynamics
    x_{k+1} = f(x_k, p)
with x in R^n and p in R^m uncertain in the box [P_lo, P_hi], find a
barrier B: R^n -> R (independent of p) such that the unsafe set is
unreachable from the initial set under EVERY admissible p in the box.

SOS conditions (Positivstellensatz form):
    -B(x) - sum_i L0_i(x) * g_init_i(x) + gamma                  is SOS in x
     B(x) - sum_i L1_{j,i}(x) * g_unsafe_{j,i}(x) - lambda_      is SOS in x
   - B(f(x, p)) + B(x)
       - sum_i Ls_i(x, p) * g_space_i(x)
       - sum_k Lp_k(x, p) * g_param_k(p)                        is SOS in (x, p)

The barrier basis lives in x only; the parameter dependence is absorbed
by Ls (in x, p) and the parameter-box multipliers Lp.

Validate-side surface (v2 parity with ct_DS_robust):
    validate_sos       -> coefficient + pointwise check
    validate_tolerance -> threshold for the combined verdict
    margin             -> forces lambda - gamma >= margin
    mosek_tol          -> forwards tighter MOSEK feasibility tolerances
    maximize_separation -> add objective to maximize lambda - gamma
"""

import time as _time

import numpy as np
import sympy as sp

import picos
from SumOfSquares import SOSProblem, poly_variable

from .generate_polynomial import generate_polynomial


def dt_DS_robust(
    b_degree, dim,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    x, f,
    p_syms=(), P_lo=(), P_hi=(),
    solver='mosek',
    gam=None, lam=None, l_degree=None,
    margin=0.0,
    mosek_tol=None,
    validate_sos=False,
    validate_tolerance=1e-8,
    maximize_separation=False,
):
    result = {'b_degree': b_degree}

    n = len(x)
    if not (len(L_initial) == dim == len(U_initial) == len(L_space) == len(U_space) == n == len(f)):
        raise ValueError("length of state arrays doesn't match dimensions!")
    m = len(p_syms)
    if not (len(P_lo) == m == len(P_hi)):
        raise ValueError("len(p_syms) and parameter-box arrays must agree")

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
            gamma = sp.symbols('gamma_robust')
            gv = prob.sym_to_var(gamma); prob.add_constraint(gv > 0)
        else:
            if gam < 0:
                raise Exception("Gamma is less than zero!")
            gamma = gam

        if lam is None:
            lambda_ = sp.symbols('lambda_robust')
            lv = prob.sym_to_var(lambda_); prob.add_constraint(lv > 0)
        else:
            if lam < 0:
                raise Exception("Lambda is less than zero!")
            lambda_ = lam

        margin_val = float(margin)
        if gam is None and lam is None:
            prob.add_constraint(lv - gv >= margin_val) if margin_val > 0 else prob.add_constraint(lv - gv > 0)
        elif gam is None:
            prob.add_constraint(lambda_ - gv >= margin_val) if margin_val > 0 else prob.add_constraint(lambda_ - gv > 0)
        elif lam is None:
            prob.add_constraint(lv - gamma >= margin_val) if margin_val > 0 else prob.add_constraint(lv - gamma > 0)
        else:
            if lam <= gam + margin_val:
                raise Exception("User defined lambda - gamma is below the requested margin!")

        if maximize_separation and gam is None and lam is None:
            prob.set_objective('max', lv - gv)
        elif maximize_separation and gam is None:
            prob.set_objective('min', gv)
        elif maximize_separation and lam is None:
            prob.set_objective('max', lv)

    except Exception:
        return {'error': 'Gamma or Lambda definition issues', 'b_degree': b_degree}

    # Discrete-time decrease condition: -B(f(x, p)) + B(x) >= sum L*g >= 0 on X x P.
    # Use Dummy substitution to evaluate B at f(x, p) safely under sympy.
    y = [sp.Dummy(f'y{i}') for i in range(n)]
    Barrier_f = Barrier.subs([(x[i], y[i]) for i in range(n)])
    Barrier_f = Barrier_f.subs([(y[i], f[i]) for i in range(n)])

    try:
        first_condition = prob.add_sos_constraint(
            -Barrier - sum(Li * gi for Li, gi in zip(L0, g0_polys)) + gamma, x)

        for j in range(n_unsafe):
            second_condition = prob.add_sos_constraint(
                Barrier - sum(Lji * gji for Lji, gji in zip(L1[j], g1_polys[j])) - lambda_, x)

        last_condition = prob.add_sos_constraint(
            -Barrier_f + Barrier
            - sum(Lsi * gi for Lsi, gi in zip(Ls, g_space))
            - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param)),
            xp)

        barrier_constraint = prob.add_sos_constraint(Barrier, x)

        for i in L0: prob.add_sos_constraint(i, x)
        for j in range(n_unsafe):
            for i in L1[j]: prob.add_sos_constraint(i, x)
        for i in Ls: prob.add_sos_constraint(i, xp)
        for k in Lp: prob.add_sos_constraint(k, xp)

    except AssertionError:
        return {'error': 'AssertionError (probably odd b_degree)', 'b_degree': b_degree}

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

    if len(barrier_constraint.get_sos_decomp().free_symbols) == 0:
        return {'error': 'barrier is scalar!', 'b_degree': b_degree}

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

    if not (result['lambda'] > result['gamma']
            and result['lambda'] > 0 and result['gamma'] > 0):
        if result['lambda'] <= result['gamma']:
            return {'error': 'lambda not greater than gamma', 'b_degree': b_degree}
        return {'error': 'numerical error on level sets', 'b_degree': b_degree}

    if validate_sos:
        from .sos_validate import (validate_problem, pointwise_validate,
                                   pointwise_verdict)
        named = [
            ('init', first_condition,
             -Barrier - sum(Li * gi for Li, gi in zip(L0, g0_polys)) + gamma,
             list(x)),
            ('step',
             last_condition,
             -Barrier_f + Barrier
             - sum(Lsi * gi for Lsi, gi in zip(Ls, g_space))
             - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param)),
             list(xp)),
            ('barrier', barrier_constraint, Barrier, list(x)),
            ('unsafe_last', second_condition,
             Barrier - sum(Li * gi for Li, gi in zip(L1[-1], g1_polys[-1])) - lambda_,
             list(x)),
        ]
        v = validate_problem(prob, named, tolerance=validate_tolerance)
        result['sos_residuals'] = {
            k: (rv if isinstance(rv, (int, float)) else rv[0])
            for k, rv in v['residuals'].items()
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
            # For discrete-time, the "Lie analogue" is f(x, p) - x; we
            # check B(f(x, p)) - B(x) <= 0 pointwise by passing
            # `dynamics_exprs` as the difference of one-step images, which
            # pointwise_validate's <grad B, f> evaluation does not match.
            # We instead just check init / unsafe pointwise and report a
            # separate "step_slack" by direct evaluation.
            pw = pointwise_validate(
                B_saved, list(x), result['gamma'], result['lambda'],
                L_initial, U_initial, unsafe_pairs, L_space, U_space,
                dynamics_exprs=None,
                p_syms=list(p_syms), p_samples=p_samples,
            )
            # Direct discrete-time step slack:
            #   sup_{x in X, p in P}  [B(f(x, p)) - B(x)]   (target <= 0)
            B_fn = sp.lambdify(list(x), B_saved, 'numpy')
            grad = None  # not used
            rng = np.random.default_rng(0)
            step_sup = -np.inf
            step_witness = None
            for p_val in (p_samples if p_samples else [None]):
                f_subs = (list(f) if p_val is None
                          else [sp.sympify(fi).subs(
                                    {p_syms[k]: float(p_val[k]) for k in range(len(p_syms))})
                                for fi in f])
                B_at_f_expr = B_saved
                yvars = [sp.Dummy(f'_y{i}') for i in range(n)]
                B_at_f_expr = B_at_f_expr.subs([(x[i], yvars[i]) for i in range(n)])
                B_at_f_expr = B_at_f_expr.subs([(yvars[i], f_subs[i]) for i in range(n)])
                step_expr = sp.expand(B_at_f_expr - B_saved)
                try:
                    step_fn = sp.lambdify(list(x), step_expr, 'numpy')
                except Exception:
                    continue
                samples = rng.uniform(L_space, U_space, size=(5000, n))
                vals = np.asarray(step_fn(*[samples[:, i] for i in range(n)]), dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    i_max = int(np.argmax(vals))
                    if float(vals[i_max]) > step_sup:
                        step_sup = float(vals[i_max])
                        step_witness = tuple(float(v) for v in samples[i_max])
            pw['step_slack'] = step_sup
            pw['step_worst_point'] = step_witness
            pw['worst_signed_slack'] = max(pw['worst_signed_slack'], step_sup)
            verdict = pointwise_verdict(pw, tolerance=validate_tolerance)
            pw['verdict'] = verdict
            result['pointwise'] = {
                'init_slack':         pw['init_slack'],
                'unsafe_slack':       pw['unsafe_slack'],
                'step_slack':         pw['step_slack'],
                'init_worst_point':   pw['init_worst_point'],
                'unsafe_worst_point': pw['unsafe_worst_point'],
                'step_worst_point':   pw['step_worst_point'],
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
