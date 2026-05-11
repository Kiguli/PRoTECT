"""
SMT verification for PRoTECT v2 barrier certificates.

The SOS solver (MOSEK / CVXOPT) is *numerical*: it solves an SDP whose
floating-point feasible point implies feasibility of the SOS programme
to within solver tolerance. That tolerance is enough for academic
soundness arguments, but it does NOT guarantee the certificate has no
counterexample under exact-arithmetic checks. This module re-checks
each safety condition with Z3 in rational arithmetic:

  (i)   forall x in Init: B(x) <= gamma
  (ii)  forall x in Unsafe_j: B(x) >= lambda      (one per region j)
  (iii) forall x in StateSpace: <grad B, f(x, p)> <= 0
                                forall p in Param

For each, the verifier asks Z3 whether the NEGATION is satisfiable. If
unsat -> the condition holds (certificate is valid for this property).
If sat -> Z3 produces a counterexample state, which is a true witness
that the SOS certificate is numerically loose.

The barrier polynomial coefficients (printed with ~3 decimal places by
SOS) are converted to rationals via ``Rational(c).limit_denominator``
so Z3 reasons exactly on the SOS-output expression. The user can then
decide:
  - Accept tolerance (the certificate is "almost valid"; acceptable
    for repeatability if the violation is below 1e-3 say).
  - Reject and re-solve with tighter MOSEK tolerance / larger margins.
"""

import os
import time

import sympy as sp


def _sym_to_z3(expr, sym_to_z3_var):
    """Recursive sympy -> z3 polynomial conversion (real arithmetic)."""
    import z3

    expr = sp.sympify(expr)
    if expr.is_Number:
        if expr.is_Integer:
            return z3.RealVal(int(expr))
        if expr.is_Rational:
            return z3.RealVal(int(expr.p)) / z3.RealVal(int(expr.q))
        if expr.is_Float:
            r = sp.Rational(float(expr)).limit_denominator(10 ** 9)
            return z3.RealVal(int(r.p)) / z3.RealVal(int(r.q))
        # Fallback: cast to float then to rational.
        try:
            r = sp.Rational(float(expr)).limit_denominator(10 ** 9)
            return z3.RealVal(int(r.p)) / z3.RealVal(int(r.q))
        except (TypeError, ValueError):
            raise ValueError(f'unsupported numeric literal: {expr}')
    if expr.is_Symbol:
        if expr not in sym_to_z3_var:
            raise KeyError(f'symbol {expr!s} not in z3 var table')
        return sym_to_z3_var[expr]
    if expr.is_Add:
        out = _sym_to_z3(expr.args[0], sym_to_z3_var)
        for a in expr.args[1:]:
            out = out + _sym_to_z3(a, sym_to_z3_var)
        return out
    if expr.is_Mul:
        out = _sym_to_z3(expr.args[0], sym_to_z3_var)
        for a in expr.args[1:]:
            out = out * _sym_to_z3(a, sym_to_z3_var)
        return out
    if expr.is_Pow:
        base = _sym_to_z3(expr.args[0], sym_to_z3_var)
        exponent = expr.args[1]
        if exponent.is_Integer and int(exponent) >= 0:
            out = z3.RealVal(1)
            for _ in range(int(exponent)):
                out = out * base
            return out
        # Fractional exponent: not supported (won't appear in SOS output).
        raise ValueError(f'non-integer exponent in barrier: {exponent}')
    raise ValueError(f'unsupported sympy node {type(expr).__name__}: {expr}')


def _z3_box_constraints(z3_vars, lo, hi):
    """Return a list of z3 inequalities encoding ``lo[i] <= var <= hi[i]``."""
    import z3
    constraints = []
    for v, l, h in zip(z3_vars, lo, hi):
        l_r = sp.Rational(float(l)).limit_denominator(10 ** 9)
        h_r = sp.Rational(float(h)).limit_denominator(10 ** 9)
        constraints.append(v >= z3.RealVal(int(l_r.p)) / z3.RealVal(int(l_r.q)))
        constraints.append(v <= z3.RealVal(int(h_r.p)) / z3.RealVal(int(h_r.q)))
    return constraints


def _check(solver_constraints, timeout_s, eval_violation=None):
    """Set up a z3 solver, add constraints, return (status, cex, violation).
    eval_violation is an optional sympy expression that, if a counter-
    example is found, gets evaluated at the cex point so we can report
    the magnitude of the violation."""
    import z3
    # QF_NRA is Z3's nonlinear-arithmetic SAT engine (Jovanovic &
    # de Moura, IJCAR 2012); it's typically faster than the default
    # solver on multivariate polynomial-inequality problems.
    s = z3.SolverFor('QF_NRA')
    s.set('timeout', int(timeout_s * 1000))
    for c in solver_constraints:
        s.add(c)
    res = s.check()
    if res == z3.unsat:
        return 'unsat', None, None
    if res == z3.sat:
        m = s.model()
        cex = {}
        for d in m:
            try:
                val = m[d]
                cex[str(d)] = float(val.as_decimal(20).rstrip('?'))
            except Exception:
                cex[str(d)] = str(m[d])
        violation = None
        if eval_violation is not None:
            try:
                # Substitute cex floats into the sympy expression.
                subs = {sp.Symbol(k): float(v) for k, v in cex.items()
                        if isinstance(v, (int, float))}
                violation = float(sp.sympify(eval_violation).subs(subs))
            except Exception:
                violation = None
        return 'sat', cex, violation
    return 'unknown', None, None


def verify_barrier(
    barrier, x_syms, dynamics,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    gamma, lambda_,
    p_syms=(), P_lo=None, P_hi=None,
    tolerance=0.0,
    timeout_s=60.0,
):
    """
    Run Z3 on each of the three safety conditions. Return a dict:
        {'initial':  ('unsat'|'sat'|'unknown', cex_or_none, t_seconds),
         'unsafe':   list[(...)],   # one per unsafe region
         'lie':      ('unsat'|'sat'|'unknown', cex_or_none, t_seconds),
         'overall':  'verified' | 'counterexample' | 'timeout' }

    tolerance > 0 relaxes the inequality strictness (accept B(x) <=
    gamma + tolerance for the initial-set check, etc). Useful when the
    SOS solver's floating-point precision puts the certificate slightly
    outside strict feasibility.
    """
    import z3

    z3_x = [z3.Real(str(s)) for s in x_syms]
    sym_to_z3 = dict(zip(x_syms, z3_x))
    z3_p = [z3.Real(str(s)) for s in p_syms]
    for s, v in zip(p_syms, z3_p):
        sym_to_z3[s] = v

    barrier_z3 = _sym_to_z3(barrier, sym_to_z3)
    gamma_r = sp.Rational(float(gamma)).limit_denominator(10 ** 9)
    lambda_r = sp.Rational(float(lambda_)).limit_denominator(10 ** 9)
    tol_r = sp.Rational(float(tolerance)).limit_denominator(10 ** 9)
    gamma_z3 = z3.RealVal(int(gamma_r.p)) / z3.RealVal(int(gamma_r.q))
    lambda_z3 = z3.RealVal(int(lambda_r.p)) / z3.RealVal(int(lambda_r.q))
    tol_z3 = z3.RealVal(int(tol_r.p)) / z3.RealVal(int(tol_r.q))

    out = {'initial': None, 'unsafe': [], 'lie': None, 'tolerance': float(tolerance)}

    barrier_sp = sp.sympify(barrier)

    # (i) Initial: cex = exists x in Init s.t. B(x) > gamma + tol.
    #     violation magnitude = B(x_cex) - gamma
    t0 = time.time()
    cs = _z3_box_constraints(z3_x, L_initial, U_initial)
    cs.append(barrier_z3 > gamma_z3 + tol_z3)
    init_viol = barrier_sp - sp.sympify(float(gamma))
    status, cex, viol = _check(cs, timeout_s, init_viol)
    out['initial'] = (status, cex, viol, round(time.time() - t0, 2))

    # (ii) Unsafe (per region): cex = exists x in Unsafe_j s.t. B(x) < lambda - tol.
    #      violation magnitude = lambda - B(x_cex)
    for j, (Lu, Uu) in enumerate(zip(L_unsafe, U_unsafe)):
        t0 = time.time()
        cs = _z3_box_constraints(z3_x, Lu, Uu)
        cs.append(barrier_z3 < lambda_z3 - tol_z3)
        viol_expr = sp.sympify(float(lambda_)) - barrier_sp
        status, cex, viol = _check(cs, timeout_s, viol_expr)
        out['unsafe'].append((status, cex, viol, round(time.time() - t0, 2)))

    # (iii) Lie: cex = exists (x, p) in StateSpace x Param
    #            s.t. <grad B, f(x, p)> > tol.
    grad = [sp.diff(barrier_sp, xi) for xi in x_syms]
    lie_expr = sum(grad[i] * sp.sympify(dynamics[i]) for i in range(len(x_syms)))
    lie_z3 = _sym_to_z3(sp.expand(lie_expr), sym_to_z3)
    t0 = time.time()
    cs = _z3_box_constraints(z3_x, L_space, U_space)
    if p_syms:
        cs += _z3_box_constraints(z3_p, P_lo, P_hi)
    cs.append(lie_z3 > tol_z3)
    status, cex, viol = _check(cs, timeout_s, lie_expr)
    out['lie'] = (status, cex, viol, round(time.time() - t0, 2))

    # Overall verdict.
    statuses = ([out['initial'][0], out['lie'][0]] +
                [u[0] for u in out['unsafe']])
    if all(s == 'unsat' for s in statuses):
        out['overall'] = 'verified'
    elif any(s == 'sat' for s in statuses):
        out['overall'] = 'counterexample'
    else:
        out['overall'] = 'timeout-or-unknown'
    return out
