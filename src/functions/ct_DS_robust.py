"""
Continuous-time deterministic-system barrier search with PARAMETER ROBUSTNESS
(PRoTECT v2 feature 7a).

Given dynamics
    x' = f(x, p)
where x in R^n is the state and p in R^m is an UNCERTAIN parameter
constrained to a box p in [p_lo, p_hi], find a barrier certificate
B: R^n -> R (independent of p) such that the unsafe set is unreachable
from the initial set under EVERY admissible parameter value.

Compared to the lifting-based encoding (treat p as an extra state with
p' = 0, search for B(x, p)):
    - The decision variable B is polynomial in x only, so its basis is
      n-dimensional rather than (n+m)-dimensional. For typical systems
      this shrinks the dominant SOS coefficient count by a factor of
      C(n+m+d, d) / C(n+d, d) where d is the barrier degree.
    - The Lie-derivative SOS constraint is over (x, p), with a fresh
      parameter-box S-procedure multiplier for each p in p_ranges.
    - The certificate is GUARANTEED valid for every p in the box, not
      just for a sampled value. This is strictly stronger than fixing p
      at a midpoint or sweeping discrete values.

The other SOS conditions (Barrier non-negativity, initial-set, and each
unsafe region) involve B(x) and the corresponding state-only g(x)
polynomials, so they remain in the n-dimensional state vector.

Math (Positivstellensatz form):

    -B(x) - sum_i L0_i(x) * g_init_i(x) + gamma                  is SOS in x
     B(x) - sum_i L1_{j,i}(x) * g_unsafe_{j,i}(x) - lambda_      is SOS in x   (per region j)
    -<dB/dx, f(x, p)>
        - sum_i Ls_i(x, p) * g_space_i(x)
        - sum_k Lp_k(x, p) * g_param_k(p)                        is SOS in (x, p)

with g_param_k(p) = (p_k - p_lo_k) * (p_hi_k - p_k).

Inputs:
    p_syms: list of sympy symbols for the parameters (length m).
    P_lo, P_hi: 1-D arrays of length m with the parameter box bounds.
    f: dynamics array, may depend symbolically on x AND p_syms.

Returns the same result dictionary shape as ct_DS.
"""

import numpy as np
import sympy as sp

import picos
from SumOfSquares import SOSProblem, poly_variable

from .generate_polynomial import generate_polynomial


def ct_DS_robust(
    b_degree, dim,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    x, f,
    p_syms, P_lo, P_hi,
    solver='mosek',
    gam=None, lam=None, l_degree=None,
    margin=0.0,
    mosek_tol=None,
    validate_sos=False,
    validate_tolerance=1e-8,
    maximize_separation=False,
    init_margin=0.0,
    unsafe_margin=0.0,
    lie_margin=0.0,
):
    result = {'b_degree': b_degree}

    # --- argument shape checks -------------------------------------------
    n = len(x)
    if not (len(L_initial) == dim == len(U_initial) == len(L_space) == len(U_space) == n == len(f)):
        raise ValueError("length of state arrays doesn't match dimensions!")

    m = len(p_syms)
    if not (len(P_lo) == m == len(P_hi)):
        raise ValueError("length of parameter arrays doesn't match number of p_syms!")

    if l_degree is None:
        l_degree = b_degree

    n_unsafe = len(L_unsafe)
    if n_unsafe != len(U_unsafe):
        raise ValueError("Unsafe regions were not defined correctly.")

    # --- state-only g polynomials (initial / unsafe / state-space) -------
    g0_polys = generate_polynomial(x, L_initial, U_initial)
    g1_polys = [generate_polynomial(x, L_unsafe[j], U_unsafe[j]) for j in range(n_unsafe)]
    g_space  = generate_polynomial(x, L_space, U_space)

    # --- parameter-box S-procedure polynomials ---------------------------
    # g_param[k] = (p_k - P_lo[k]) * (P_hi[k] - p_k);  >= 0 iff p_k in [P_lo, P_hi].
    g_param = [(p_syms[k] - P_lo[k]) * (P_hi[k] - p_syms[k]) for k in range(m)]

    # Combined variable list for the Lie-derivative SOS constraint.
    xp = list(x) + list(p_syms)

    prob = SOSProblem()

    try:
        # B(x) only -- key difference from the lifting approach.
        Barrier = poly_variable('Barrier', x, b_degree)

        # Lagrangians for initial set: in x.
        L0 = [poly_variable('L0_' + str(i + 1), x, l_degree) for i in range(n)]

        # Lagrangians for each unsafe region: in x.
        L1 = [[poly_variable('La_' + str(j) + '_' + str(i + 1), x, l_degree)
               for i in range(n)]
              for j in range(n_unsafe)]

        # Lagrangians for state space: in (x, p) so they can absorb
        # any p-dependent residuals when multiplying the Lie derivative.
        Ls = [poly_variable('Ls_' + str(i + 1), xp, l_degree) for i in range(n)]

        # Lagrangians for parameter box: in (x, p).
        Lp = [poly_variable('Lp_' + str(k + 1), xp, l_degree) for k in range(m)]

        # gamma, lambda decision variables (scalars).
        if gam is None:
            gamma = sp.symbols('gamma_robust')
            gv = prob.sym_to_var(gamma)
            prob.add_constraint(gv > 0)
        else:
            if gam < 0:
                raise Exception("Gamma is less than zero!")
            gamma = gam

        if lam is None:
            lambda_ = sp.symbols('lambda_robust')
            lv = prob.sym_to_var(lambda_)
            prob.add_constraint(lv > 0)
        else:
            if lam < 0:
                raise Exception("Lambda is less than zero!")
            lambda_ = lam

        # `margin` forces lambda - gamma >= margin (default 0 keeps the
        # original strict-inequality behaviour). Positive margin gives Z3
        # a real separation gap between init-side and unsafe-side level
        # sets, which is needed for the certificate to verify in exact
        # rational arithmetic.
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
                    "User defined lambda - gamma is below the requested margin!")

        # If requested, set an objective to MAXIMIZE the separation
        # lambda - gamma. Without this, MOSEK just finds the cheapest
        # feasible (gamma, lambda) pair which can leave them numerically
        # coincident -- visually indistinguishable in a barrier-figure
        # render. The maximisation is bounded by the SOS constraints
        # (B <= gamma on X_0, B >= lambda on X_u) given a finite barrier.
        if maximize_separation and gam is None and lam is None:
            prob.set_objective('max', lv - gv)
        elif maximize_separation and gam is None:
            prob.set_objective('min', gv)
        elif maximize_separation and lam is None:
            prob.set_objective('max', lv)

    except Exception:
        return {'error': 'Gamma or Lambda definition issues', 'b_degree': b_degree}

    # --- Lie derivative (polynomial in (x, p)) ---------------------------
    LieDeriv = np.array([sp.diff(Barrier, xi) for xi in x])
    Barrier_f = np.sum(LieDeriv * f)

    # Per-condition strict-positivity margins. The Positivstellensatz is
    # rigorous in exact arithmetic: -B - sum L_i g_i + gamma is SOS implies
    # B <= gamma on X_0 (and analogously for the unsafe and Lie
    # conditions). With MOSEK's coefficient tolerance epsilon ~ 1e-8,
    # the polynomial identity holds only up to epsilon, which at large
    # basis values on the boundary can give pointwise drift of order
    # epsilon * max_basis_value (~1e-5 for degree-4 polynomials on
    # [-3, 3]). Each per-condition margin below shifts the SOS expression
    # by delta > 0 so the asserted polynomial is forced to be >= delta
    # everywhere on R^n. The certificate then has a rigorous pointwise
    # margin of at least delta - epsilon * max_basis_value on the
    # asserted set -- positive as long as delta exceeds the solver's
    # amplified noise floor.
    init_delta   = float(init_margin)
    unsafe_delta = float(unsafe_margin)
    lie_delta    = float(lie_margin)

    try:
        # Initial-set SOS: in x.
        L0_g0 = [Li * gi for Li, gi in zip(L0, g0_polys)]
        first_condition = prob.add_sos_constraint(
            -Barrier - sum(L0_g0) + gamma - init_delta, x)

        # Unsafe-region SOS: in x, one per region.
        for j in range(n_unsafe):
            L1j_g1j = [Lji * gji for Lji, gji in zip(L1[j], g1_polys[j])]
            second_condition = prob.add_sos_constraint(
                Barrier - sum(L1j_g1j) - lambda_ - unsafe_delta, x)

        # Lie-derivative SOS: in (x, p), with both state-space and
        # parameter-box S-procedure multipliers.
        Ls_gspace = [Lsi * gi for Lsi, gi in zip(Ls, g_space)]
        Lp_gparam = [Lpk * gpk for Lpk, gpk in zip(Lp, g_param)]
        last_condition = prob.add_sos_constraint(
            -Barrier_f - sum(Ls_gspace) - sum(Lp_gparam) - lie_delta,
            xp,
        )

        # Barrier itself SOS in x.
        barrier_constraint = prob.add_sos_constraint(Barrier, x)

        # All multipliers must be SOS over their respective variable lists.
        for i in L0:
            prob.add_sos_constraint(i, x)
        for j in range(n_unsafe):
            for i in L1[j]:
                prob.add_sos_constraint(i, x)
        for i in Ls:
            prob.add_sos_constraint(i, xp)
        for k in Lp:
            prob.add_sos_constraint(k, xp)

    except AssertionError:
        return {'error': 'AssertionError (probably odd b_degree)', 'b_degree': b_degree}

    # --- solve -----------------------------------------------------------
    import time as _time
    _solve_t0 = _time.time()
    try:
        # Tight MOSEK feasibility / optimality tolerances. PICOS forwards
        # `mosek_params` keys to mosek directly; the names follow the
        # Mosek MSK_DPAR / MSK_IPAR enumeration.
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
                'solve_time': _time.time() - _solve_t0}
    except Exception:
        return {'error': 'Solver Exception', 'b_degree': b_degree,
                'solve_time': _time.time() - _solve_t0}
    _solve_time = _time.time() - _solve_t0
    result['solve_time'] = _solve_time

    # --- results ---------------------------------------------------------
    if len(barrier_constraint.get_sos_decomp().free_symbols) == 0:
        return {'error': 'barrier is scalar!', 'b_degree': b_degree}

    if (len(barrier_constraint.get_sos_decomp()) > 0 and
        len(first_condition.get_sos_decomp()) > 0 and
        len(second_condition.get_sos_decomp()) > 0 and
        len(last_condition.get_sos_decomp()) > 0):
        # Save with FULL coefficient precision rather than the default
        # 3 d.p. rounding from get_sos_decomp() -- the rounding is fine
        # for well-scaled benchmarks but loses ~10^-3 of every coefficient,
        # which after multiplying by stiff dynamics (1e7 in ROBE25/3) and
        # large basis values at the state-space boundary causes the saved
        # barrier to violate Lie <= 0 by orders of magnitude. precision=20
        # asks `round_sympy_expr` for 20 d.p., effectively no rounding.
        try:
            result['barrier'] = sum(barrier_constraint.get_sos_decomp(precision=20))
        except Exception:
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

    if not (result['lambda'] > result['gamma']
            and result['lambda'] > 0 and result['gamma'] > 0):
        if result['lambda'] <= result['gamma']:
            return {'error': 'lambda not greater than gamma', 'b_degree': b_degree}
        return {'error': 'numerical error on level sets e.g. negative value',
                'b_degree': b_degree}

    # --- post-solve numerical validation -------------------------------
    if validate_sos:
        from .sos_validate import (validate_problem, pointwise_validate,
                                   pointwise_verdict)
        named = [
            ('init',
             first_condition,
             -Barrier - sum(Li * gi for Li, gi in zip(L0, g0_polys)) + gamma - init_delta,
             list(x)),
            ('lie',
             last_condition,
             -Barrier_f
             - sum(Lsi * gi for Lsi, gi in zip(Ls, g_space))
             - sum(Lpk * gpk for Lpk, gpk in zip(Lp, g_param))
             - lie_delta,
             list(xp)),
            ('barrier', barrier_constraint, Barrier, list(x)),
            ('unsafe_last', second_condition,
             Barrier - sum(Li * gi for Li, gi in zip(L1[-1], g1_polys[-1])) - lambda_ - unsafe_delta,
             list(x)),
        ]
        v = validate_problem(prob, named, tolerance=validate_tolerance)
        result['sos_residuals'] = {
            k: (rv if isinstance(rv, (int, float)) else rv[0])
            for k, rv in v['residuals'].items()
        }
        result['sos_status'] = v['status']
        result['sos_overall'] = v['overall']

        # ---- POINTWISE check on the closed asserted boxes ----
        # Coefficient-space residuals (above) measure the SOS
        # decomposition's faithfulness to the asserted polynomial; they
        # do NOT detect S-procedure-multiplier absorption that allows
        # B(x) < lambda inside X_u at solver tolerance. The pointwise
        # check below samples the boxes directly and evaluates B (and
        # the Lie derivative across parameter samples) at corners +
        # interior points. The combined verdict 'overall' below is
        # PASS only if BOTH the coefficient check and the pointwise
        # check pass.
        try:
            B_saved = result.get('barrier', Barrier)
            unsafe_pairs = list(zip(L_unsafe, U_unsafe))
            # Build parameter samples across the box (midpoint + corners)
            # when the system has parameters.
            if len(p_syms):
                import numpy as _np, itertools as _it
                p_lo = _np.asarray(P_lo, float); p_hi = _np.asarray(P_hi, float)
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
                dynamics_exprs=list(f),
                p_syms=list(p_syms), p_samples=p_samples,
            )
            verdict = pointwise_verdict(pw, tolerance=validate_tolerance)
            result['pointwise'] = {
                'init_slack':          pw['init_slack'],
                'unsafe_slack':        pw['unsafe_slack'],
                'lie_slack':           pw['lie_slack'],
                'init_worst_point':    pw['init_worst_point'],
                'unsafe_worst_point':  pw['unsafe_worst_point'],
                'lie_worst_point':     pw['lie_worst_point'],
                'verdict':             verdict,
            }
            # Combined overall verdict: certificate is reported clean only
            # if BOTH coefficient and pointwise checks pass.
            old_overall = result['sos_overall']
            if old_overall == 'clean' and verdict == 'pass':
                result['sos_overall'] = 'clean'
            elif old_overall in ('clean', 'warning') and verdict == 'warn':
                result['sos_overall'] = 'warning'
            else:
                result['sos_overall'] = 'fail'
        except Exception as exc:
            result['pointwise'] = {'error': f'pointwise eval failed: {exc}'}
        # NOTE: validation failure no longer sets `error`. The barrier is
        # accepted; sos_overall ('clean'/'warning'/'fail') and the
        # residuals are reported separately so the runner / CSV can flag
        # falsified-post-hoc certificates while NOT triggering a costly
        # retry. Per the design discussion: only swap to CVXOPT when
        # MOSEK couldn't produce a barrier at all.
    return result
