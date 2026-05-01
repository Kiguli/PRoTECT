"""
Polynomial Pade approximants for non-polynomial dynamics terms
(PRoTECT v2 feature 1b).

A Pade approximant of order [m/n] for a function f at the expansion
point x = a is the unique rational R(x) = P_m(x) / Q_n(x) that matches
the Taylor series of f at a up to order m + n. For PRoTECT we:

  * Build P_m(x) / Q_n(x) symbolically.
  * Multiply through by Q_n(x) so the resulting expression is polynomial.
  * Optionally introduce an auxiliary state q satisfying q * Q_n(x) = P_m(x),
    bounded by the residual error of the truncation, so the SOS programme
    can substitute f(x) -> q with q in [m, M] (just like the relaxation
    registry's sqrt / inv_power flow).

This module gives the Pade construction; the auxiliary-state plumbing is
provided in src/functions/relaxations.py for sqrt / inv_power; for
arbitrary f, the user picks the tightening strategy.

Implementation notes:
- We rely on sympy's series expansion to get the Taylor coefficients,
  then solve the Pade linear system. For small (m, n) this is exact.
- The error bound ``f - R`` is computed numerically over a user-provided
  evaluation grid (for tightness diagnostics).
"""

import sympy as sp


def pade_coefficients(taylor_coeffs, m, n):
    """
    Given a Taylor series ``f(x) = sum_k c_k x^k`` (coefficients
    ``c_0, ..., c_{m+n}``), return ``(p_coeffs, q_coeffs)`` such that
    ``f(x) ~ P_m(x) / Q_n(x)`` and ``Q_n(0) = 1``.

    Solves the linear system from the Pade definition:
        sum_{i=0}^{n} q_i c_{k-i} = p_k         for k = 0..m
        sum_{i=0}^{n} q_i c_{k-i} = 0           for k = m+1..m+n
        q_0 = 1

    Returns
    -------
    p_coeffs : list of length m+1
    q_coeffs : list of length n+1, with q_coeffs[0] == 1
    """
    if len(taylor_coeffs) < m + n + 1:
        raise ValueError("need at least m+n+1 Taylor coefficients")

    c = [sp.sympify(ci) for ci in taylor_coeffs[: m + n + 1]]
    q_syms = [sp.Symbol(f'q_{i}') for i in range(n + 1)]
    p_syms = [sp.Symbol(f'p_{i}') for i in range(m + 1)]

    eqs = []
    eqs.append(q_syms[0] - 1)

    # q . c convolution = p (at orders 0..m), 0 (at orders m+1..m+n)
    for k in range(m + n + 1):
        s = 0
        for i in range(min(k, n) + 1):
            s = s + q_syms[i] * c[k - i]
        if k <= m:
            eqs.append(s - p_syms[k])
        else:
            eqs.append(s)

    sol = sp.solve(eqs, q_syms + p_syms, dict=True)
    if not sol:
        raise RuntimeError("Pade system is singular (try a different (m, n))")
    sol = sol[0]
    return [sol[p] for p in p_syms], [sol[q] for q in q_syms]


def pade_expression(f_taylor_around, var, m, n, expansion_point=0):
    """
    Build the Pade rational approximant ``P_m(var) / Q_n(var)`` from a
    Sympy callable ``f_taylor_around`` returning a Taylor series at
    ``expansion_point`` with at least m+n+1 terms.

    Parameters
    ----------
    f_taylor_around : sympy expression in `var`
    var : sympy symbol
    m, n : Pade orders
    expansion_point : the point about which to expand (default 0)

    Returns
    -------
    P, Q : sympy polynomials in `var`
    """
    series = sp.series(f_taylor_around, var, expansion_point, m + n + 2).removeO()
    poly = sp.Poly(series, var)
    coeffs = list(reversed(poly.all_coeffs()))  # ascending order
    while len(coeffs) < m + n + 1:
        coeffs.append(sp.Integer(0))

    p, q = pade_coefficients(coeffs, m, n)
    P = sum(p[i] * (var - expansion_point) ** i for i in range(m + 1))
    Q = sum(q[i] * (var - expansion_point) ** i for i in range(n + 1))
    return sp.expand(P), sp.expand(Q)


def polynomialise_via_pade(f, var, q_sym, m, n, expansion_point=0):
    """
    Replace f(var) with a fresh symbol ``q_sym``, plus the EQUALITY
    multiplier ``q_sym * Q(var) - P(var) == 0`` so SOS picks up a
    Pade-tight relaxation.

    Returns
    -------
    expr_substitute : ``q_sym``
    equality        : sympy expression that should equal 0
    P, Q            : the Pade numerator and denominator
    """
    P, Q = pade_expression(f, var, m, n, expansion_point=expansion_point)
    return q_sym, q_sym * Q - P, P, Q
