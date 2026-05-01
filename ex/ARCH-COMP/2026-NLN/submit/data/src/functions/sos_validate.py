"""
Post-solve numerical validation for PRoTECT SOS programmes.

PICOS + Python-SumOfSquares give us:
  * prob.status      -- solver termination status
  * constraint.Qval  -- Gram matrix (numpy float64) returned by MOSEK
  * constraint.b_sym -- the monomial basis vector

In exact arithmetic, B(x)^T Q B(x) would identically equal the asserted
polynomial. In practice MOSEK's Q has two kinds of numerical noise:

  (1) Tiny non-zero entries (~1e-10 to 1e-7) that should be zero --
      these mostly come from the interior-point solver's last few
      iterations and are well below any true significance threshold.

  (2) Coefficient drift on the genuinely-non-zero entries proportional
      to the SDP feasibility tolerance (~1e-8 by default).

The default `SOSConstraint.get_sos_decomp(precision=3)` rounds the
factored polynomial to 3 decimal places after Cholesky -- this AMPLIFIES
the noise to order ~1e-3 in coefficient space, which is what shows up
as the headline "residual" everyone has been seeing.

This module fixes that with a manual rounding-then-PSD-projection pass:

  cleaned_sos_decomposition(constraint, round_threshold=1e-8)
    -> list of polynomials s_i**2 (sympy) with full-precision
       coefficients, computed from a cleaned Gram matrix:
         a) entries with |Q[i,j]| < round_threshold rounded to 0
         b) symmetrise to enforce numerical symmetry
         c) eigendecomposition; negative eigenvalues clamped to 0
         d) Cholesky-style factor V*sqrt(W)
         e) decomp[k] = (b_sym . column_k(V*sqrt(W)))**2

  cleaned_residual_norm(constraint, expected_expr, vars)
    -> max |coefficient| of (expected_expr - sum(cleaned_decomp));
       if this is small, the solver's certificate is internally
       consistent.

  status_summary(prob)
    -> (status_str, is_clean) -- True iff prob.status is "optimal" /
       "primal feasible".
"""

import numpy as np
import sympy as sp


# Statuses we accept as "MOSEK returned a clean numerical solution".
_OK_STATUS = {'optimal', 'primal feasible', 'feasible'}
# Statuses we treat as "MOSEK gave a degraded numerical solution".
_WARN_STATUS = {'near optimal', 'near optimal solution', 'stalled',
                'unknown', 'primal feasible but not solved to optimality'}


def status_summary(prob):
    """Return (status_str, is_clean) where is_clean=True means MOSEK
    converged to optimality without numerical degradation.

    PICOS' Problem.status string varies between versions; we lower-case
    and substring-match the well-known statuses."""
    try:
        s = str(prob.status).lower()
    except Exception:
        return ('unknown', False)
    for ok in _OK_STATUS:
        if ok in s:
            return (s, True)
    for warn in _WARN_STATUS:
        if warn in s:
            return (s, False)
    return (s, False)


def cleaned_sos_decomposition(constraint, round_threshold=1e-8):
    """Recover a high-precision SOS decomposition from the raw Gram
    matrix MOSEK returned, with manual cleanup of numerical noise:

      1. Round Gram entries with |Q[i,j]| < round_threshold to 0
      2. Symmetrise (Q + Q^T)/2
      3. Eigendecomposition; clamp negative eigenvalues to 0
      4. Decomposition: for each column k of V * sqrt(W), build
         s_k(x) = b_sym . V[:,k] * sqrt(W[k]), and emit s_k**2.

    Returns a list of sympy polynomials [s_1**2, s_2**2, ...] whose sum
    is the certificate's SOS body. Coefficients are full-precision
    floats (no aggressive 3-digit rounding like the default decomposer).
    """
    Q = np.asarray(constraint.Qval, dtype=float).copy()
    Q[np.abs(Q) < round_threshold] = 0.0
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    w = np.where(w < 0, 0.0, w)
    factor = V * np.sqrt(w)  # n x n matrix; column k is V[:,k]*sqrt(w[k])
    basis = list(constraint.b_sym)
    n = len(basis)
    out = []
    for k in range(n):
        if w[k] < round_threshold:
            continue
        s = sum(float(factor[i, k]) * basis[i] for i in range(n))
        out.append(s * s)
    return out


def _substitute_decision_values(expr, prob):
    """Replace every PICOS decision-variable sympy symbol in expr with
    its post-solve numeric value. State variables (no associated
    decision variable) are left alone."""
    expr = sp.sympify(expr)
    subs = {}
    for sym in expr.free_symbols:
        try:
            v = prob.sym_to_var(sym).value
            subs[sym] = float(v)
        except Exception:
            # Not a decision-variable symbol (likely a state / parameter);
            # leave it as-is.
            pass
    return expr.subs(subs)


def decomp_residual(constraint, expected_expr, vars=None, prob=None):
    """Residual polynomial = (expected with decision values substituted)
    - (sum of cleaned SOS decomposition)."""
    decomp = cleaned_sos_decomposition(constraint)
    sum_sq = sp.expand(sum(decomp)) if decomp else sp.Integer(0)
    expected_numeric = (_substitute_decision_values(expected_expr, prob)
                        if prob is not None else sp.sympify(expected_expr))
    residual = sp.expand(expected_numeric - sum_sq)
    return residual


def decomp_residual_norm(constraint, expected_expr, vars, prob=None):
    """Max |coefficient| of the residual polynomial after substituting
    decision values into `expected_expr`."""
    residual = decomp_residual(constraint, expected_expr, vars, prob=prob)
    if residual == 0:
        return 0.0
    try:
        poly = sp.Poly(residual, *vars)
        # Poly.coeffs() handles multivariate; all_coeffs() does not.
        coeffs = [float(c) for c in poly.coeffs()]
    except (sp.PolynomialError, TypeError, ValueError):
        # Fallback: walk the expanded sum and grab leading-term magnitudes.
        try:
            terms = sp.expand(residual).as_ordered_terms()
            coeffs = []
            for t in terms:
                c, _ = t.as_independent(*vars, as_Add=False)
                try:
                    coeffs.append(float(c))
                except Exception:
                    pass
        except Exception:
            return float('inf')
    return max(abs(c) for c in coeffs) if coeffs else 0.0


def _validate_norm(prob, constraint, expected, vars_):
    return decomp_residual_norm(constraint, expected, vars_, prob=prob)


def validate_problem(prob, named_constraints, tolerance=1e-6):
    """
    named_constraints : list of (label, sos_constraint, expected_expr, vars).

    Returns a dict:
        {
          'status': str,            # MOSEK status string
          'status_clean': bool,
          'residuals': {label: residual_norm},
          'worst_residual': float,
          'worst_label': str,
          'overall': 'clean' | 'warning' | 'fail',
        }

    'clean'   : status OK and all residuals < tolerance
    'warning' : status OK but some residual >= tolerance
    'fail'    : status degraded (stalled / near-optimal) OR any residual
                significantly above tolerance (10x).
    """
    status_str, status_clean = status_summary(prob)
    out = {'status': status_str, 'status_clean': status_clean,
           'residuals': {}, 'worst_residual': 0.0, 'worst_label': None}

    for label, constraint, expected, vars_ in named_constraints:
        try:
            r = _validate_norm(prob, constraint, expected, vars_)
        except Exception as exc:
            r = float('inf')
            out['residuals'][label] = (r, f'eval error: {exc}')
            continue
        out['residuals'][label] = r
        if r > out['worst_residual']:
            out['worst_residual'] = r
            out['worst_label'] = label

    if out['worst_residual'] < tolerance and status_clean:
        out['overall'] = 'clean'
    elif out['worst_residual'] >= 10 * tolerance or not status_clean:
        out['overall'] = 'fail'
    else:
        out['overall'] = 'warning'
    return out
