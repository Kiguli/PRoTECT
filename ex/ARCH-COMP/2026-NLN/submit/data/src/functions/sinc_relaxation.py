"""
Polynomial relaxation of sin / cos terms via sinc and an S-procedure-style
auxiliary state.

Idea: sin(a) = sinc(a) * a, and sinc(a) = sin(a)/a is bounded on any closed
interval. Introduce a fresh state q = sinc(a) constrained to [m, M] (so that
{q : a in I} contains the true sinc image). Substituting sin(a) = q*a turns
non-polynomial dynamics into polynomial-in-(x, q) dynamics; q gets the
trivial flow q' = 0 and contributes box-shaped initial / state / unsafe
constraints handled natively by ct_DS / dt_DS.

For cos the analogous identity is
    cos(a) = 1 - 2 * sinc(a/2)^2 * (a/2)^2 = 1 - r * a^2 / 2
with r := sinc(a/2)^2 in [m_c, 1] over the same range.

Bounds on a closed interval [-A, A]:
    sinc range:  [sin(A)/A, 1]  if A < pi  (sinc is even, monotonically
                                            decreasing on [0, pi] from 1 to
                                            sin(pi)/pi = 0)
    sinc(a/2)^2: [(sin(A/2)/(A/2))^2, 1]
"""

import math


def sinc_bounds(angle_max):
    """
    Tight (analytic) lower bound on sinc(a) = sin(a)/a for |a| <= angle_max,
    and the corresponding bound on sinc(a/2)^2.

    Returns (sinc_lo, sinc_sq_half_lo). Both upper bounds are 1.

    angle_max must satisfy 0 < angle_max < pi.
    """
    if not (0.0 < angle_max < math.pi):
        raise ValueError(
            "sinc_bounds requires 0 < angle_max < pi; got %r" % angle_max
        )
    sinc_lo = math.sin(angle_max) / angle_max
    half = angle_max / 2.0
    sinc_half = math.sin(half) / half
    return sinc_lo, sinc_half ** 2


def replace_sin(angle_expr, q_sym):
    """
    Return the polynomial substitute q_sym * angle_expr for sin(angle_expr).
    Caller must add q_sym to the state vector with bounds (sinc_lo, 1) from
    sinc_bounds(angle_max), and add the trivial flow q_sym' = 0.
    """
    return q_sym * angle_expr


def replace_cos(angle_expr, r_sym):
    """
    Return the polynomial substitute 1 - r_sym * angle_expr**2 / 2 for
    cos(angle_expr). Caller must add r_sym to the state vector with bounds
    (sinc_sq_half_lo, 1) from sinc_bounds(angle_max), and add the trivial
    flow r_sym' = 0.
    """
    return 1 - r_sym * angle_expr**2 / 2
