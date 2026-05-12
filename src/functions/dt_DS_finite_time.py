"""
Discrete-time deterministic finite-time-horizon barrier (PRoTECT v2).

Searches a time-augmented barrier
    B(x, k) = sum_{j=0..K} k^j B_j(x)
certifying that no trajectory of x_{k+1} = f(x_k, p) starting in X_0
enters X_u during k in {0, 1, ..., N}, for every p in [P_lo, P_hi].

SOS conditions:
    -B(x, 0) - sum_i L0_i(x) g_init_i(x) + gamma          is SOS in x
     B(x, k) - sum_i Lu_{j,i}(x, k) g_unsafe_{j,i}(x)
             - Lt_j(x, k) g_t(k) - lambda                 is SOS in (x, k)  per region j
   - B(f(x, p), k+1) + B(x, k)
       - sum_i Ls_i(x, k, p) g_space_i(x)
       - Lt_lie(x, k, p) g_t(k)
       - sum_q Lp_q(x, k, p) g_param_q(p)                 is SOS in (x, k, p)

with g_t(k) = k * (N - k), g_param_q(p) = (p_q - P_lo[q]) (P_hi[q] - p_q).

Pointwise check uses corner + interior sampling of X_0, X_u, X x [0, N].
"""

import time as _time
import numpy as np
import sympy as sp

import picos
from SumOfSquares import SOSProblem, poly_variable

from .generate_polynomial import generate_polynomial


def dt_DS_finite_time(
    b_degree, time_orders,
    x, f,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    N_steps,
    time_sym=None,
    p_syms=(), P_lo=(), P_hi=(),
    margin=0.0,
    solver='mosek',
    mosek_tol=None,
    gam=None, lam=None, l_degree=None,
    validate_sos=False,
    validate_tolerance=1e-8,
    dim=None,
    init_margin=0.0,
    unsafe_margin=0.0,
    step_margin=0.0,
):
    n = len(x)
    if len(f) != n:
        raise ValueError("len(f) must equal len(x)")
    if dim is not None and dim != n:
        raise ValueError("len(x) and dim must agree")
    m = len(p_syms)
    if not (len(P_lo) == m == len(P_hi)):
        raise ValueError("p_syms, P_lo, P_hi must agree in length")

    if time_sym is None:
        time_sym = sp.Symbol('k_finite')
    k = time_sym
    xk  = list(x) + [k]
    xkp = list(x) + [k] + list(p_syms)

    g_initial = list(generate_polynomial(x, L_initial, U_initial))
    g_space   = list(generate_polynomial(x, L_space, U_space))
    g_unsafe  = [list(generate_polynomial(x, Lj, Uj))
                 for Lj, Uj in zip(L_unsafe, U_unsafe)]
    g_time   = k * (sp.sympify(N_steps) - k)
    g_param  = [(p_syms[q] - sp.sympify(P_lo[q])) * (sp.sympify(P_hi[q]) - p_syms[q])
                for q in range(m)]

    if l_degree is None:
        l_degree = b_degree

    result = {'b_degree': b_degree, 'time_orders': time_orders, 'N_steps': int(N_steps)}
    prob = SOSProblem()

    try:
        Bk = [poly_variable(f'Bj_{j}', x, b_degree) for j in range(time_orders + 1)]
        Barrier = sum(Bk[j] * k**j for j in range(time_orders + 1))

        L_init = [poly_variable(f'L0i_{i+1}', x, l_degree)
                  for i in range(len(g_initial))]
        L_unsafe_per_region = [
            [poly_variable(f'Lu_{j}_{i+1}', xk, l_degree)
             for i in range(len(g_unsafe[j]))]
            for j in range(len(g_unsafe))
        ]
        L_t_unsafe = [poly_variable(f'Lt_unsafe_{j+1}', xk, l_degree)
                      for j in range(len(g_unsafe))]
        Ls = [poly_variable(f'Ls_{i+1}', xkp, l_degree)
              for i in range(len(g_space))]
        Lt_lie = poly_variable('Lt_lie', xkp, l_degree)
        Lp = [poly_variable(f'Lp_{q+1}', xkp, l_degree) for q in range(m)]

        if gam is None:
            gamma = sp.symbols('gamma_ft_dt')
            gv = prob.sym_to_var(gamma); prob.add_constraint(gv > 0)
        else:
            if gam < 0:
                raise Exception("Gamma is less than zero!")
            gamma = gam
        if lam is None:
            lambda_ = sp.symbols('lambda_ft_dt')
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
                raise Exception("User-defined lambda - gamma below the requested margin!")

    except Exception:
        return {'error': 'Gamma/Lambda definition issues', 'b_degree': b_degree,
                'time_orders': time_orders}

    # Discrete-time step: B(f(x, p), k+1) - B(x, k) <= 0.
    y = [sp.Dummy(f'_y{i}') for i in range(n)]
    Barrier_at_k = Barrier
    Barrier_at_k1_fx = Barrier.subs(k, k + 1)
    Barrier_at_k1_fx = Barrier_at_k1_fx.subs([(x[i], y[i]) for i in range(n)])
    Barrier_at_k1_fx = Barrier_at_k1_fx.subs([(y[i], f[i]) for i in range(n)])

    Barrier_at_0 = Barrier.subs(k, 0)

    # Per-condition strict-positivity margins.
    init_delta = float(init_margin)
    unsafe_delta = float(unsafe_margin)
    step_delta = float(step_margin)

    try:
        first_condition = prob.add_sos_constraint(
            -Barrier_at_0
            - sum(Li * gi for Li, gi in zip(L_init, g_initial))
            + gamma - init_delta,
            x,
        )

        last_unsafe = None
        for j in range(len(g_unsafe)):
            terms = Barrier
            terms = terms - sum(Li * gi for Li, gi in zip(L_unsafe_per_region[j], g_unsafe[j]))
            terms = terms - L_t_unsafe[j] * g_time
            terms = terms - lambda_ - unsafe_delta
            last_unsafe = prob.add_sos_constraint(terms, xk)

        last_condition = prob.add_sos_constraint(
            -Barrier_at_k1_fx + Barrier_at_k
            - sum(Li * gi for Li, gi in zip(Ls, g_space))
            - Lt_lie * g_time
            - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param))
            - step_delta,
            xkp,
        )

        for Li in L_init: prob.add_sos_constraint(Li, x)
        for region in L_unsafe_per_region:
            for Li in region: prob.add_sos_constraint(Li, xk)
        for Lt in L_t_unsafe: prob.add_sos_constraint(Lt, xk)
        for Li in Ls: prob.add_sos_constraint(Li, xkp)
        prob.add_sos_constraint(Lt_lie, xkp)
        for Lpq in Lp: prob.add_sos_constraint(Lpq, xkp)

        barrier_constraint = prob.add_sos_constraint(Barrier, xk)

    except AssertionError:
        return {'error': 'AssertionError (probably odd b_degree or time_orders)',
                'b_degree': b_degree, 'time_orders': time_orders}

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
                'time_orders': time_orders, 'solve_time': _time.time() - _t0}
    except Exception:
        return {'error': 'Solver Exception', 'b_degree': b_degree,
                'time_orders': time_orders, 'solve_time': _time.time() - _t0}
    result['solve_time'] = _time.time() - _t0

    if (len(barrier_constraint.get_sos_decomp()) > 0
        and len(first_condition.get_sos_decomp()) > 0
        and last_unsafe is not None
        and len(last_unsafe.get_sos_decomp()) > 0
        and len(last_condition.get_sos_decomp()) > 0):
        try:
            result['barrier'] = sum(barrier_constraint.get_sos_decomp(precision=20))
        except Exception:
            result['barrier'] = sum(barrier_constraint.get_sos_decomp())
    else:
        return {'error': 'constraints not SOS', 'b_degree': b_degree,
                'time_orders': time_orders}

    result['gamma']  = float(gv) if gam is None else gam
    result['lambda'] = float(lv) if lam is None else lam

    if not (result['lambda'] > result['gamma']
            and result['lambda'] > 0 and result['gamma'] > 0):
        if result['lambda'] <= result['gamma']:
            return {'error': 'lambda not greater than gamma',
                    'b_degree': b_degree, 'time_orders': time_orders}
        return {'error': 'numerical error on level sets',
                'b_degree': b_degree, 'time_orders': time_orders}

    if validate_sos:
        from .sos_validate import validate_problem
        named = [
            ('init', first_condition,
             -Barrier_at_0
             - sum(Li * gi for Li, gi in zip(L_init, g_initial))
             + gamma - init_delta, list(x)),
            ('step', last_condition,
             -Barrier_at_k1_fx + Barrier_at_k
             - sum(Li * gi for Li, gi in zip(Ls, g_space))
             - Lt_lie * g_time
             - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param))
             - step_delta,
             list(xkp)),
            ('barrier', barrier_constraint, Barrier, list(xk)),
            ('unsafe_last', last_unsafe,
             Barrier
             - sum(Li * gi for Li, gi in zip(
                 L_unsafe_per_region[-1], g_unsafe[-1]))
             - L_t_unsafe[-1] * g_time
             - lambda_ - unsafe_delta, list(xk)),
        ]
        v = validate_problem(prob, named, tolerance=validate_tolerance)
        result['sos_residuals'] = {
            kk: (rv if isinstance(rv, (int, float)) else rv[0])
            for kk, rv in v['residuals'].items()
        }
        result['sos_status']  = v['status']
        result['sos_overall'] = v['overall']

    return result
