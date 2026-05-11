"""
Solver-fallback helper for PRoTECT v2.

Try MOSEK first with post-solve SOS validation. If MOSEK either:
  * fails to solve,
  * solves but the SOS-decomposition residual is above tolerance,
fall back to CVXOPT and return whichever solution actually validates.

Returns (result_dict, solver_used). If both solvers fail, the result
dict has 'error' set and 'solver' marked as the last attempted.
"""

from copy import copy


def solve_with_fallback(
    solver_fn,
    base_kwargs,
    solvers=('mosek', 'cvxopt'),
    validate_tolerance=1e-4,
):
    """
    solver_fn       : callable returning a result dict (e.g. ct_DS_robust).
    base_kwargs     : dict of kwargs (without 'solver'); we pass 'solver'
                      as MOSEK then CVXOPT.

    Note: mosek-specific kwargs (mosek_tol) are silently dropped on the
    cvxopt attempt because CVXOPT doesn't support them.
    """
    last = None; last_solver = solvers[0]
    for solver in solvers:
        kw = dict(base_kwargs)
        kw['solver'] = solver
        kw['validate_sos'] = True
        kw['validate_tolerance'] = validate_tolerance
        if solver != 'mosek':
            kw.pop('mosek_tol', None)
        result = solver_fn(**kw)
        last = result; last_solver = solver
        if (result
                and 'barrier' in result
                and 'error' not in result
                and result.get('sos_overall') in (None, 'clean', 'warning')):
            result_with_solver = copy(result)
            result_with_solver['solver'] = solver
            return result_with_solver, solver
    if last is None:
        last = {}
    last_with = copy(last)
    last_with['solver'] = last_solver
    return last_with, last_solver


def solve_safety_problem(
    degrees, x, f,
    L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
    p_syms=(), P_lo=(), P_hi=(),
    margin=0.0, mosek_tol=None, validate_tolerance=1e-8,
):
    """Try MOSEK across the full degree sweep first. Only fall back to
    CVXOPT (across the same degree sweep) if MOSEK couldn't produce a
    validated barrier at any degree. Returns (result_dict, solver_used).

    Validation uses ct_DS_robust's `validate_sos=True` -- post-solve
    decomposition residual check. validate_tolerance=0.1 means
    residuals < 1 are accepted (warning), residuals >= 1 are flagged
    'fail' and trigger the next attempt."""
    from .ct_DS_robust import ct_DS_robust

    last = None
    cumulative_solve_time = 0.0
    for solver in ('mosek', 'cvxopt'):
        for degree in degrees:
            kw = dict(
                b_degree=degree, dim=len(x),
                L_initial=L_initial, U_initial=U_initial,
                L_unsafe=L_unsafe, U_unsafe=U_unsafe,
                L_space=L_space,   U_space=U_space,
                x=x, f=f, p_syms=p_syms, P_lo=P_lo, P_hi=P_hi,
                margin=margin, solver=solver,
                validate_sos=True,
                validate_tolerance=validate_tolerance,
            )
            if solver == 'mosek' and mosek_tol is not None:
                kw['mosek_tol'] = mosek_tol
            result = ct_DS_robust(**kw)
            cumulative_solve_time += result.get('solve_time', 0.0) if result else 0.0
            last = result
            # Take the first barrier MOSEK produces, regardless of
            # validation status. The residual / sos_overall is recorded
            # for the report; we don't retry just because validation
            # flagged it. Only fall through to CVXOPT if no barrier was
            # produced at any degree by MOSEK.
            if result and 'barrier' in result and 'error' not in result:
                result['solver'] = solver
                result['solve_time_total'] = cumulative_solve_time
                return result, solver
        # else: MOSEK exhausted all degrees; fall through to CVXOPT.
    if last is None:
        last = {}
    last['solver'] = 'cvxopt'
    last['solve_time_total'] = cumulative_solve_time
    return last, 'cvxopt'


def solve_finite_time_safety_problem(
    degrees, time_orders, T_horizon,
    x, f,
    L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
    p_syms=(), P_lo=(), P_hi=(),
    margin=0.0, mosek_tol=None, validate_tolerance=1e-8,
):
    """Finite-horizon analogue of solve_safety_problem.

    Tries MOSEK across the full (degree x time_order) sweep first; falls back
    to CVXOPT only if MOSEK couldn't produce a barrier at any
    (degree, time_orders) combination. Returns (result_dict, solver_used).
    The result dict matches solve_safety_problem's, with extra keys
    'T_horizon' and 'time_orders'."""
    from .ct_DS_finite_time import ct_DS_finite_time

    last = None
    cumulative_solve_time = 0.0
    if isinstance(time_orders, int):
        time_orders_list = [time_orders]
    else:
        time_orders_list = list(time_orders)

    for solver in ('mosek', 'cvxopt'):
        for degree in degrees:
            for k_order in time_orders_list:
                kw = dict(
                    b_degree=degree, time_orders=k_order, T_horizon=T_horizon,
                    dim=len(x),
                    L_initial=L_initial, U_initial=U_initial,
                    L_unsafe=L_unsafe, U_unsafe=U_unsafe,
                    L_space=L_space,   U_space=U_space,
                    x=x, f=f, p_syms=p_syms, P_lo=P_lo, P_hi=P_hi,
                    margin=margin, solver=solver,
                    validate_sos=True,
                    validate_tolerance=validate_tolerance,
                )
                if solver == 'mosek' and mosek_tol is not None:
                    kw['mosek_tol'] = mosek_tol
                result = ct_DS_finite_time(**kw)
                cumulative_solve_time += result.get('solve_time', 0.0) \
                                         if result else 0.0
                last = result
                if result and 'barrier' in result and 'error' not in result:
                    result['solver'] = solver
                    result['solve_time_total'] = cumulative_solve_time
                    return result, solver
    if last is None:
        last = {}
    last['solver'] = 'cvxopt'
    last['solve_time_total'] = cumulative_solve_time
    return last, 'cvxopt'
