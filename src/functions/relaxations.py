"""
Polynomial relaxations of non-polynomial dynamics terms (PRoTECT v2).

Generalises the sinc(x) trick into a small registry of name-keyed
relaxations for sin / cos / tan / exp / log / sqrt / 1-over-r^k. Each
relaxation:

  1. Returns a polynomial substitute for the non-polynomial term, with a
     fresh auxiliary variable q (or several) standing in for the
     non-polynomial part.
  2. Tells the caller the box bounds [m, M] that q must satisfy over the
     given input range (so that the substitute is sound, i.e. there
     exists a value of q in [m, M] making the substitute equal to the
     true value of the term).

The caller appends each q to the SOS state vector with the trivial flow
q' = 0 (continuous-time) or q_{k+1} = q_k (discrete-time). The PRoTECT
SOS pipeline then handles the box constraint via the standard
S-procedure that already exists for the state-space envelope.

Math sketch (per relaxation):

  sin(a)   = q * a                with q in [sinc(A), 1]
  cos(a)   = 1 - r * a^2 / 2      with r in [sinc(A/2)^2, 1]
  tan(a)   = q * a                with q in [1, tan(A)/A]   (|a| < pi/2)
  exp(a)   = q                    with q in [exp(a_lo), exp(a_hi)]
  log(a)   = q                    with q in [log(a_lo), log(a_hi)]
  sqrt(a)  = q                    with q in [sqrt(a_lo), sqrt(a_hi)],
                                  plus q^2 = a as an equality multiplier
  1/a^k    = q                    with q in [1/a_hi^k, 1/a_lo^k],
                                  plus q * a^k = 1 as an equality multiplier

A is the absolute bound on the angle (sin/cos/tan); a_lo, a_hi are the
range bounds on a (positive for log/sqrt/inv).

Conventions:
  * All relaxations return a `Relaxation` namedtuple-style object with:
        expr        : sympy expression substituted for the original term
        aux_vars    : list of (sympy.Symbol, lo, hi) tuples
        equalities  : list of sympy expressions that should equal zero
                      (plain-equality S-procedure multipliers; empty for
                      the box-only relaxations sin / cos / tan / exp / log)
  * Equality multipliers are added to ALL SOS conditions with a free
    polynomial Lagrangian (no sign restriction). PRoTECT's existing
    pipeline doesn't support equalities yet -- handling those is part of
    the relaxation registry's deeper integration (see TODO at end of file).
"""

import math
from collections import namedtuple

import sympy as sp


Relaxation = namedtuple('Relaxation', ['expr', 'aux_vars', 'equalities'])
# expr        : sympy.Expr      polynomial substitute
# aux_vars    : list of tuples  [(sym, lo, hi), ...]
# equalities  : list of sympy   [eq_expr, ...] each meant as `eq_expr == 0`


# ---------------------------------------------------------------------------
# sin / cos / tan
# ---------------------------------------------------------------------------

def relax_sin(angle_expr, q_sym, angle_max):
    """
    sin(angle_expr) -> q_sym * angle_expr,
    with q_sym in [sin(A)/A, 1] for |angle_expr| <= A = angle_max.
    Requires 0 < angle_max < pi.
    """
    if not (0.0 < angle_max < math.pi):
        raise ValueError("relax_sin requires 0 < angle_max < pi")
    lo = math.sin(angle_max) / angle_max
    return Relaxation(
        expr=q_sym * angle_expr,
        aux_vars=[(q_sym, lo, 1.0)],
        equalities=[],
    )


def relax_cos(angle_expr, r_sym, angle_max):
    """
    cos(angle_expr) -> 1 - r_sym * angle_expr**2 / 2,
    with r_sym in [(sin(A/2)/(A/2))^2, 1].
    Identity: cos(a) = 1 - 2*sin^2(a/2) = 1 - 2*(sinc(a/2)*(a/2))^2
                     = 1 - r * a^2 / 2  with r := sinc(a/2)^2.
    Requires 0 < angle_max < pi.
    """
    if not (0.0 < angle_max < math.pi):
        raise ValueError("relax_cos requires 0 < angle_max < pi")
    half = angle_max / 2.0
    sinc_half = math.sin(half) / half
    lo = sinc_half ** 2
    return Relaxation(
        expr=1 - r_sym * angle_expr**2 / 2,
        aux_vars=[(r_sym, lo, 1.0)],
        equalities=[],
    )


def relax_tan(angle_expr, q_sym, angle_max):
    """
    tan(angle_expr) -> q_sym * angle_expr,
    with q_sym in [1, tan(A)/A] for |angle_expr| <= A = angle_max.
    Tan grows superlinearly so the lower bound on the slope is 1 (at 0).
    Requires 0 < angle_max < pi/2.
    """
    if not (0.0 < angle_max < math.pi / 2):
        raise ValueError("relax_tan requires 0 < angle_max < pi/2")
    hi = math.tan(angle_max) / angle_max
    return Relaxation(
        expr=q_sym * angle_expr,
        aux_vars=[(q_sym, 1.0, hi)],
        equalities=[],
    )


# ---------------------------------------------------------------------------
# exp / log
# ---------------------------------------------------------------------------

def relax_exp(arg_expr, q_sym, arg_lo, arg_hi):
    """
    exp(arg_expr) -> q_sym, with q_sym in [exp(arg_lo), exp(arg_hi)].
    Sound under the assumption arg_expr in [arg_lo, arg_hi]; the caller
    must ensure that constraint holds (it usually follows from the
    state-space envelope or from monotonicity of arg_expr in x).
    """
    if arg_lo > arg_hi:
        raise ValueError("relax_exp: arg_lo must be <= arg_hi")
    return Relaxation(
        expr=q_sym,
        aux_vars=[(q_sym, math.exp(arg_lo), math.exp(arg_hi))],
        equalities=[],
    )


def relax_log(arg_expr, q_sym, arg_lo, arg_hi):
    """
    log(arg_expr) -> q_sym, with q_sym in [log(arg_lo), log(arg_hi)].
    Requires arg_lo > 0.
    """
    if arg_lo <= 0:
        raise ValueError("relax_log requires arg_lo > 0")
    if arg_lo > arg_hi:
        raise ValueError("relax_log: arg_lo must be <= arg_hi")
    return Relaxation(
        expr=q_sym,
        aux_vars=[(q_sym, math.log(arg_lo), math.log(arg_hi))],
        equalities=[],
    )


# ---------------------------------------------------------------------------
# sqrt and 1/r^k -- these need an EQUALITY constraint to be tight, so they
# return a non-empty `equalities` list. See TODO at the bottom of the file.
# ---------------------------------------------------------------------------

def relax_sqrt(arg_expr, q_sym, arg_lo, arg_hi):
    """
    sqrt(arg_expr) -> q_sym, with q_sym in [sqrt(arg_lo), sqrt(arg_hi)],
    plus the equality multiplier q_sym**2 - arg_expr == 0 to keep the
    relaxation tight (otherwise q is only constrained by the box, which
    drops information about which value of q corresponds to which arg).
    Requires arg_lo >= 0.
    """
    if arg_lo < 0:
        raise ValueError("relax_sqrt requires arg_lo >= 0")
    if arg_lo > arg_hi:
        raise ValueError("relax_sqrt: arg_lo must be <= arg_hi")
    return Relaxation(
        expr=q_sym,
        aux_vars=[(q_sym, math.sqrt(arg_lo), math.sqrt(arg_hi))],
        equalities=[q_sym**2 - arg_expr],
    )


def relax_inv_power(arg_expr, q_sym, arg_lo, arg_hi, k=1):
    """
    1 / arg_expr**k -> q_sym, with q_sym in [1/arg_hi**k, 1/arg_lo**k],
    plus the equality multiplier q_sym * arg_expr**k - 1 == 0.
    Requires arg_lo > 0 and k >= 1 integer.
    """
    if arg_lo <= 0:
        raise ValueError("relax_inv_power requires arg_lo > 0")
    if not (isinstance(k, int) and k >= 1):
        raise ValueError("relax_inv_power requires integer k >= 1")
    if arg_lo > arg_hi:
        raise ValueError("relax_inv_power: arg_lo must be <= arg_hi")
    return Relaxation(
        expr=q_sym,
        aux_vars=[(q_sym, 1.0 / arg_hi**k, 1.0 / arg_lo**k)],
        equalities=[q_sym * arg_expr**k - 1],
    )


# ---------------------------------------------------------------------------
# Registry-style dispatch (for clean call sites in benchmarks).
# ---------------------------------------------------------------------------

RELAXATIONS = {
    'sin':       relax_sin,
    'cos':       relax_cos,
    'tan':       relax_tan,
    'exp':       relax_exp,
    'log':       relax_log,
    'sqrt':      relax_sqrt,
    'inv_power': relax_inv_power,
}


def relax(name, *args, **kwargs):
    """Look up a named relaxation and apply it."""
    if name not in RELAXATIONS:
        raise KeyError(
            "Unknown relaxation %r; available: %s"
            % (name, sorted(RELAXATIONS))
        )
    return RELAXATIONS[name](*args, **kwargs)


# ---------------------------------------------------------------------------
# Composite / convenience helpers
# ---------------------------------------------------------------------------

def relax_sin_cos_pair(angle_expr, q_sin_sym, q_cos_sym, angle_max):
    """
    Return both sin and cos relaxations for the same angle, sharing the
    angle-range A. Useful in TRAF22 / TSPS25 where sin and cos of the
    same heading appear together. Returns (sin_relax, cos_relax).
    """
    return relax_sin(angle_expr, q_sin_sym, angle_max), \
           relax_cos(angle_expr, q_cos_sym, angle_max)


def stack_relaxations(*relaxations):
    """
    Combine a list of Relaxation objects into a single (aux_vars,
    equalities) bundle. Returns (aux_vars, equalities).
    """
    aux = []
    eqs = []
    for r in relaxations:
        aux.extend(r.aux_vars)
        eqs.extend(r.equalities)
    return aux, eqs


# ---------------------------------------------------------------------------
# TODO (v2 follow-on, not in this file):
#
#   1. Plumb `equalities` through ct_DS / ct_SS / dt_DS / dt_SS so that
#      relax_sqrt and relax_inv_power are usable directly. The current
#      PRoTECT SOS pipeline only consumes inequality (g >= 0)
#      constraints; adding equality multipliers means accepting an
#      unconstrained polynomial Lagrangian for each `eq_expr` and adding
#      `lambda_eq * eq_expr` to the relevant SOS sums.
#
#   2. Wire `aux_vars` into the standard state-vector construction so
#      benchmark scripts don't have to manually extend L_space / U_space.
#      A `build_problem(states, dynamics, relaxations=...)` helper would
#      do this.
#
#   3. For sqrt and inv_power specifically, also support a direct
#      "Padé-style" relaxation that avoids the equality multiplier (at
#      the cost of looser bounds), giving users a tightness/feasibility
#      trade-off knob.
# ---------------------------------------------------------------------------
