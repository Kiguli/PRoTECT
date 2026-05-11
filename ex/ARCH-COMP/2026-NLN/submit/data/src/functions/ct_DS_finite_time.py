"""
Finite-time-horizon barrier search (PRoTECT v2).

Given dynamics
    x' = f(x, p),       x in R^n,   p in [P_lo, P_hi]  (uncertain, optional)
this solver constructs a time-augmented barrier
    B(x, t) = sum_{k=0..K} t^k * B_k(x)
certifying that no trajectory starting in X_0 enters any unsafe region X_u
during the bounded horizon t in [0, T], for EVERY admissible parameter p
in the supplied box.

SOS conditions (Positivstellensatz form):

    -B(x, 0) - sum_i L0_i(x) * g_init_i(x)               + gamma     is SOS in x
     B(x, t) - sum_i Lu_{j,i}(x,t) * g_unsafe_{j,i}(x)
             - Lt_{j}(x,t)        * g_t(t)               - lambda    is SOS in (x, t)   per unsafe region j
   -(dB/dt + <dB/dx, f(x, p)>)
             - sum_i Ls_i(x,t,p)  * g_space_i(x)
             - Lt_lie(x,t,p)      * g_t(t)
             - sum_k Lp_k(x,t,p)  * g_param_k(p)                     is SOS in (x, t, p)

with
    g_t(t)        = t * (T - t)                  (>= 0 iff t in [0, T])
    g_param_k(p)  = (p_k - P_lo[k]) * (P_hi[k] - p_k)  (>= 0 iff p_k in box)

The barrier B(x, t) is independent of p (parameter-robust pattern); only the
Lie SOS sees p, and the parameter-box S-procedure absorbs the dependence.

Returns the same result dictionary shape as ct_DS_robust, with additional
keys 'T_horizon' and 'time_orders'.
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
    p_syms=(), P_lo=(), P_hi=(),
    margin=0.0,
    solver='mosek',
    mosek_tol=None,
    gam=None, lam=None, l_degree=None,
    validate_sos=False,
    validate_tolerance=1e-4,
    dim=None,
):
    """
    See module docstring.

    Inputs:
        b_degree, time_orders : spatial polynomial degree of each B_k and
                                the highest power of t in the time series.
        x, f                  : state symbols and dynamics expressions
                                (length n; f may depend on p_syms).
        initial_polys, ...    : either explicit non-negativity polynomials
                                or box bounds (L_*, U_*) which are converted
                                with generate_polynomial.
        T_horizon             : positive float; the horizon length.
        p_syms, P_lo, P_hi    : uncertain parameter symbols and box. Pass
                                empty tuples (default) for the non-robust
                                case.
        margin                : forces lambda - gamma >= margin (>0 gives a
                                strict separation gap useful for SMT).
        mosek_tol             : if set, forwards tighter MOSEK feasibility
                                tolerances.
        validate_sos          : run the post-solve numerical validator and
                                annotate the result with 'sos_overall' /
                                'sos_residuals' / 'sos_status'.

    Returns a dict with keys 'barrier', 'gamma', 'lambda', 'b_degree',
    'time_orders', 'T_horizon', 'solve_time', plus the validator keys if
    requested.  On failure, returns {'error': str, ...}.
    """
    n = len(x)
    if len(f) != n:
        raise ValueError("len(f) must equal len(x)")
    if dim is not None and dim != n:
        raise ValueError("len(x) and `dim` must agree")
    m = len(p_syms)
    if not (len(P_lo) == m == len(P_hi)):
        raise ValueError("len(p_syms), len(P_lo), len(P_hi) must agree")

    # --- variable lists ----------------------------------------------------
    if time_sym is None:
        time_sym = sp.Symbol('t_finite')
    t = time_sym
    xt = list(x) + [t]                 # SOS variables for (B, unsafe)
    xtp = list(x) + [t] + list(p_syms) # SOS variables for the Lie condition

    # --- non-negativity polynomials --------------------------------------
    g_initial = ([sp.sympify(g) for g in initial_polys]
                 if initial_polys is not None
                 else list(generate_polynomial(x, L_initial, U_initial)))
    g_space = ([sp.sympify(g) for g in space_polys]
               if space_polys is not None
               else list(generate_polynomial(x, L_space, U_space)))
    g_unsafe = ([[sp.sympify(g) for g in r] for r in unsafe_regions]
                if unsafe_regions is not None
                else [list(generate_polynomial(x, Lj, Uj))
                      for Lj, Uj in zip(L_unsafe, U_unsafe)])
    g_time = t * (sp.sympify(T_horizon) - t)
    g_param = [(p_syms[k] - sp.sympify(P_lo[k]))
                * (sp.sympify(P_hi[k]) - p_syms[k]) for k in range(m)]

    if l_degree is None:
        l_degree = b_degree

    result = {'b_degree': b_degree, 'time_orders': time_orders,
              'T_horizon': float(T_horizon)}
    prob = SOSProblem()

    try:
        # --- time-varying barrier ----------------------------------------
        Bk_polys = [poly_variable(f'Bk_{k}', x, b_degree)
                    for k in range(time_orders + 1)]
        Barrier = sum(Bk_polys[k] * t**k for k in range(time_orders + 1))

        # --- Lagrangians --------------------------------------------------
        L_init = [poly_variable(f'L0i_{i+1}', x, l_degree)
                  for i in range(len(g_initial))]
        L_unsafe_per_region = [
            [poly_variable(f'Lu_{j}_{i+1}', xt, l_degree)
             for i in range(len(g_unsafe[j]))]
            for j in range(len(g_unsafe))
        ]
        L_t_unsafe = [poly_variable(f'Lt_unsafe_{j+1}', xt, l_degree)
                      for j in range(len(g_unsafe))]
        Ls = [poly_variable(f'Ls_{i+1}', xtp, l_degree)
              for i in range(len(g_space))]
        Lt_lie = poly_variable('Lt_lie', xtp, l_degree)
        Lp = [poly_variable(f'Lp_{k+1}', xtp, l_degree) for k in range(m)]

        # --- gamma, lambda decision variables ----------------------------
        if gam is None:
            gamma = sp.symbols('gamma_ft')
            gv = prob.sym_to_var(gamma); prob.add_constraint(gv > 0)
        else:
            if gam < 0:
                raise Exception("Gamma is less than zero!")
            gamma = gam
        if lam is None:
            lambda_ = sp.symbols('lambda_ft')
            lv = prob.sym_to_var(lambda_); prob.add_constraint(lv > 0)
        else:
            if lam < 0:
                raise Exception("Lambda is less than zero!")
            lambda_ = lam

        margin_val = float(margin)
        if gam is None and lam is None:
            if margin_val > 0:
                prob.add_constraint(lv - gv >= margin_val)
            else:
                prob.add_constraint(lv - gv > 0)
        elif gam is None:
            if margin_val > 0:
                prob.add_constraint(lambda_ - gv >= margin_val)
            else:
                prob.add_constraint(lambda_ - gv > 0)
        elif lam is None:
            if margin_val > 0:
                prob.add_constraint(lv - gamma >= margin_val)
            else:
                prob.add_constraint(lv - gamma > 0)
        else:
            if lam <= gam + margin_val:
                raise Exception(
                    "User-defined lambda - gamma is below the requested margin!")

    except Exception:
        return {'error': 'Gamma or Lambda definition issues',
                'b_degree': b_degree, 'time_orders': time_orders}

    # --- Lie derivative dB/dt + <grad_x B, f(x, p)> --------------------
    dBdt = sp.diff(Barrier, t)
    grad_x = np.array([sp.diff(Barrier, xi) for xi in x])
    Lie = dBdt + np.sum(grad_x * np.array(f))

    Barrier_at_0 = Barrier.subs(t, 0)

    try:
        # 1) Initial set (SOS in x): -B(x,0) - L0.g0 + gamma.
        first_condition = prob.add_sos_constraint(
            -Barrier_at_0
            - sum(Li * gi for Li, gi in zip(L_init, g_initial))
            + gamma,
            x,
        )

        # 2) Unsafe set per region (SOS in (x, t)): B - Lu.gu - Lt_u.g_t - lambda.
        last_unsafe = None
        for j in range(len(g_unsafe)):
            terms = Barrier
            terms = terms - sum(
                Li * gi for Li, gi in zip(L_unsafe_per_region[j], g_unsafe[j]))
            terms = terms - L_t_unsafe[j] * g_time
            terms = terms - lambda_
            last_unsafe = prob.add_sos_constraint(terms, xt)

        # 3) Lie derivative (SOS in (x, t, p)): -Lie - Ls.gs - Lt.g_t - Lp.gp.
        last_condition = prob.add_sos_constraint(
            -Lie
            - sum(Li * gi for Li, gi in zip(Ls, g_space))
            - Lt_lie * g_time
            - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param)),
            xtp,
        )

        # 4) Multipliers must be SOS over their variable lists.
        for Li in L_init:
            prob.add_sos_constraint(Li, x)
        for region in L_unsafe_per_region:
            for Li in region:
                prob.add_sos_constraint(Li, xt)
        for Lt in L_t_unsafe:
            prob.add_sos_constraint(Lt, xt)
        for Li in Ls:
            prob.add_sos_constraint(Li, xtp)
        prob.add_sos_constraint(Lt_lie, xtp)
        for Lpk in Lp:
            prob.add_sos_constraint(Lpk, xtp)

        # 5) Barrier itself SOS in (x, t).
        barrier_constraint = prob.add_sos_constraint(Barrier, xt)

    except AssertionError:
        return {'error': 'AssertionError (probably odd b_degree)',
                'b_degree': b_degree, 'time_orders': time_orders}

    # --- solve -----------------------------------------------------------
    import time as _time
    _solve_t0 = _time.time()
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
                'time_orders': time_orders,
                'solve_time': _time.time() - _solve_t0}
    except Exception:
        return {'error': 'Solver Exception', 'b_degree': b_degree,
                'time_orders': time_orders,
                'solve_time': _time.time() - _solve_t0}
    _solve_time = _time.time() - _solve_t0
    result['solve_time'] = _solve_time

    # --- result extraction ----------------------------------------------
    if (len(barrier_constraint.get_sos_decomp()) > 0
        and len(first_condition.get_sos_decomp()) > 0
        and last_unsafe is not None
        and len(last_unsafe.get_sos_decomp()) > 0
        and len(last_condition.get_sos_decomp()) > 0):
        result['barrier'] = sum(barrier_constraint.get_sos_decomp())
    else:
        return {'error': 'constraints not SOS',
                'b_degree': b_degree, 'time_orders': time_orders}

    result['gamma']  = float(gv) if gam is None else gam
    result['lambda'] = float(lv) if lam is None else lam

    if not (result['lambda'] > result['gamma']
            and result['lambda'] > 0 and result['gamma'] > 0):
        if result['lambda'] <= result['gamma']:
            return {'error': 'lambda not greater than gamma',
                    'b_degree': b_degree, 'time_orders': time_orders}
        return {'error': 'numerical error on level sets',
                'b_degree': b_degree, 'time_orders': time_orders}

    # --- post-solve numerical validation -------------------------------
    if validate_sos:
        from .sos_validate import validate_problem
        named = [
            ('init',
             first_condition,
             -Barrier_at_0
             - sum(Li * gi for Li, gi in zip(L_init, g_initial))
             + gamma,
             list(x)),
            ('lie',
             last_condition,
             -Lie
             - sum(Li * gi for Li, gi in zip(Ls, g_space))
             - Lt_lie * g_time
             - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param)),
             list(xtp)),
            ('barrier', barrier_constraint, Barrier, list(xt)),
            ('unsafe_last', last_unsafe,
             Barrier
             - sum(Li * gi for Li, gi in zip(
                 L_unsafe_per_region[-1], g_unsafe[-1]))
             - L_t_unsafe[-1] * g_time
             - lambda_,
             list(xt)),
        ]
        v = validate_problem(prob, named, tolerance=validate_tolerance)
        result['sos_residuals'] = {
            k: (rv if isinstance(rv, (int, float)) else rv[0])
            for k, rv in v['residuals'].items()
        }
        result['sos_status']  = v['status']
        result['sos_overall'] = v['overall']

    return result
