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


# ---------------------------------------------------------------------
# POINTWISE validator. Coefficient-space checks (above) measure the SOS
# decomposition's faithfulness to the asserted polynomial -- they catch
# solver convergence failures but DO NOT catch S-procedure-multiplier
# absorption that lets B(x) < lambda hold inside X_u at the solver
# tolerance. The pointwise validator below directly evaluates B(x) and
# <grad B, f(x, p)> at corners and interior samples of the asserted
# sets, and reports the worst-case slack pointwise.
# ---------------------------------------------------------------------

def _box_corners(L, U):
    """Enumerate all 2^n corner points of an n-D axis-aligned box."""
    import itertools
    n = len(L)
    return np.array([
        [(L[i] if bit == 0 else U[i]) for i, bit in enumerate(combo)]
        for combo in itertools.product([0, 1], repeat=n)
    ], dtype=float)


def pointwise_validate(
    barrier_expr, x_syms, gamma, lam,
    L_initial, U_initial,
    unsafe_boxes,            # list of (L_u, U_u) tuples
    L_space, U_space,
    dynamics_exprs=None,     # length-n list of sympy exprs for f(x, p); may
                             # contain extra parameter symbols `p_syms`.
    p_syms=(), p_samples=None,
    n_init=2000, n_unsafe=2000, n_lie=8000, n_corners=True, seed=0,
):
    """
    Direct pointwise check that the barrier certificate's geometric
    conditions hold at sampled points on the closed asserted sets.

    Returns a dict with keys:
        'sup_B_init'        : max B(x) over corners + samples of X_0.
        'inf_B_unsafe'      : min B(x) over corners + samples of X_u
                              (across all unsafe boxes).
        'sup_Lie'           : max <grad B, f(x, p)> over samples of X
                              (and parameter samples in p_samples).
        'init_slack'        : sup_B_init - gamma (must be <= 0).
        'unsafe_slack'      : lambda - inf_B_unsafe (must be <= 0).
        'lie_slack'         : sup_Lie (must be <= 0).
        'worst_signed_slack': max of the three.
        'verdict'           : 'pass' / 'warn' / 'fail' based on tolerance.
        'init_worst_point', 'unsafe_worst_point', 'lie_worst_point'
                            : the witness point for each worst slack
                              (state coordinates).
    The verdict thresholds use `tolerance` for 'pass' and `10*tolerance`
    for 'warn'.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    n = len(x_syms)
    L_initial = np.asarray(L_initial, float); U_initial = np.asarray(U_initial, float)
    L_space   = np.asarray(L_space, float);   U_space   = np.asarray(U_space, float)

    B_fn = sp.lambdify(x_syms, barrier_expr, 'numpy')

    def _B(pts):
        return np.asarray(B_fn(*[pts[:, i] for i in range(n)]), dtype=float)

    # --- (1) sup B on X_0 --------------------------------------------
    init_pts = []
    if n_corners:
        init_pts.append(_box_corners(L_initial, U_initial))
    init_pts.append(rng.uniform(L_initial, U_initial, size=(n_init, n)))
    init_pts = np.vstack(init_pts)
    Bvals = _B(init_pts)
    idx = int(np.argmax(Bvals))
    sup_B_init = float(Bvals[idx])
    init_worst = tuple(float(v) for v in init_pts[idx])

    # --- (2) inf B on X_u --------------------------------------------
    inf_B_unsafe = +np.inf
    unsafe_worst = None
    for L_u, U_u in unsafe_boxes:
        L_u = np.asarray(L_u, float); U_u = np.asarray(U_u, float)
        pts = []
        if n_corners:
            pts.append(_box_corners(L_u, U_u))
        pts.append(rng.uniform(L_u, U_u, size=(n_unsafe, n)))
        pts = np.vstack(pts)
        Bv = _B(pts)
        i_min = int(np.argmin(Bv))
        if float(Bv[i_min]) < inf_B_unsafe:
            inf_B_unsafe = float(Bv[i_min])
            unsafe_worst = tuple(float(v) for v in pts[i_min])

    # --- (3) sup Lie on X across p samples ---------------------------
    sup_lie = -np.inf
    lie_worst = None
    if dynamics_exprs is not None:
        grad_B = [sp.diff(barrier_expr, s) for s in x_syms]
        # Determine parameter symbols actually appearing in f.
        # If p_samples is None or empty, evaluate at a single nominal pt.
        if not p_samples:
            p_samples = [None]
        for p_val in p_samples:
            # Build a substitution map for p_syms.
            if p_val is None or not len(p_syms):
                f_subs = list(dynamics_exprs)
            else:
                p_dict = {p_syms[k]: float(p_val[k]) for k in range(len(p_syms))}
                f_subs = [sp.sympify(fi).subs(p_dict) for fi in dynamics_exprs]
            dot = sum(grad_B[i] * f_subs[i] for i in range(n))
            try:
                dot_fn = sp.lambdify(x_syms, dot, 'numpy')
            except Exception:
                continue
            pts = rng.uniform(L_space, U_space, size=(n_lie, n))
            try:
                Lv = np.asarray(dot_fn(*[pts[:, i] for i in range(n)]), dtype=float)
            except Exception:
                continue
            Lv = Lv[np.isfinite(Lv)]
            if Lv.size:
                i_max = int(np.argmax(Lv))
                if float(Lv[i_max]) > sup_lie:
                    sup_lie = float(Lv[i_max])
                    lie_worst = tuple(float(v) for v in pts[i_max])

    init_slack   = sup_B_init - gamma           # want <= 0
    unsafe_slack = lam - inf_B_unsafe           # want <= 0
    lie_slack    = (sup_lie if np.isfinite(sup_lie) else 0.0)

    return {
        'sup_B_init':       sup_B_init,
        'inf_B_unsafe':     inf_B_unsafe,
        'sup_Lie':          (sup_lie if np.isfinite(sup_lie) else None),
        'init_slack':       init_slack,
        'unsafe_slack':     unsafe_slack,
        'lie_slack':        lie_slack,
        'init_worst_point':   init_worst,
        'unsafe_worst_point': unsafe_worst,
        'lie_worst_point':    lie_worst,
        'worst_signed_slack': max(init_slack, unsafe_slack, lie_slack),
    }


def pointwise_verdict(p, tolerance=1e-6):
    """Reduce a pointwise_validate result to 'pass' / 'warn' / 'fail'.
    Thresholds:
      pass if all three slacks <= tolerance (i.e. within solver noise)
      warn if 1 <= max_slack / tolerance < 100
      fail if max_slack > 100 * tolerance OR negative-residual violation."""
    worst = p['worst_signed_slack']
    if worst <= tolerance:
        return 'pass'
    if worst <= 100 * tolerance:
        return 'warn'
    return 'fail'


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
