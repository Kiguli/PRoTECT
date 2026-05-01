"""
Non-box set descriptions for PRoTECT v2 (features 2a, 2b, 2c).

The current PRoTECT pipeline accepts axis-aligned boxes for initial,
unsafe, and state-space regions via ``L_lo, L_hi`` arrays. v2 adds three
generalisations, each emitting a list of polynomial inequalities g_i(x)
suitable for direct use as Positivstellensatz multipliers:

  2a. Polytope (H-representation): {x : A x <= b}
      One linear polynomial per row of A.

  2b. Quadratic / sub-level set: {x : g(x) <= 0} for g any polynomial.
      Single polynomial inequality (we flip sign so the convention
      "all listed polynomials >= 0 inside the set" stays uniform).

  2c. Union of polynomial sub-level sets:
      {x : g_1(x) >= 0} U {x : g_2(x) >= 0} U ...
      Returns a list of lists, one inner list per region. The downstream
      SOS programme adds one Lagrangian per inner list.

All helpers mirror the convention of the existing
``generate_polynomial`` (returns a list of polynomial inequalities all
of which are >= 0 INSIDE the set).
"""

from collections.abc import Sequence

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# 2a -- Polytope via H-representation
# ---------------------------------------------------------------------------

def polytope_inequalities(x, A, b):
    """
    Encode the polytope ``{x : A x <= b}`` as a list of polynomial
    inequalities ``[b_i - sum_j A_ij x_j]`` (each >= 0 inside the set).

    Parameters
    ----------
    x : sequence of sympy symbols, length n
    A : array-like, shape (m, n)
    b : array-like, shape (m,)

    Returns
    -------
    list of sympy expressions, length m
    """
    A = np.asarray(A)
    b = np.asarray(b)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if len(x) != n:
        raise ValueError("len(x) must match A.shape[1]")
    if b.shape != (m,):
        raise ValueError("b must have shape (A.shape[0],)")
    return [
        sp.sympify(b[i]) - sum(sp.sympify(A[i, j]) * x[j] for j in range(n))
        for i in range(m)
    ]


# ---------------------------------------------------------------------------
# 2b -- Quadratic / arbitrary-polynomial sub-level set
# ---------------------------------------------------------------------------

def sublevel_set(g, sense='le'):
    """
    Encode ``{x : g(x) <op> 0}`` as a polynomial inequality list with
    each entry >= 0 INSIDE the set.

    sense='le' (default)  -> {x : g(x) <= 0}  ->  return [-g]
    sense='ge'            -> {x : g(x) >= 0}  ->  return [g]

    Returns a single-element list to match the
    ``generate_polynomial``-style convention used downstream.
    """
    if sense == 'le':
        return [-sp.sympify(g)]
    if sense == 'ge':
        return [sp.sympify(g)]
    raise ValueError("sense must be 'le' or 'ge'")


def quadratic_form_set(x, Q, c, d, sense='le'):
    """
    Convenience helper: encode the quadratic-form set
        {x : x^T Q x + c^T x + d <op> 0}
    in inequality form. Useful for ellipsoids, balls, and quadric
    halfspaces (e.g. line-of-sight cones approximated by quadratic
    relaxation).
    """
    Q = np.asarray(Q)
    c = np.asarray(c)
    n = len(x)
    if Q.shape != (n, n):
        raise ValueError("Q must be (n, n)")
    if c.shape != (n,):
        raise ValueError("c must be (n,)")
    g = sum(sp.sympify(Q[i, j]) * x[i] * x[j] for i in range(n) for j in range(n))
    g = g + sum(sp.sympify(c[i]) * x[i] for i in range(n))
    g = g + sp.sympify(d)
    return sublevel_set(g, sense=sense)


# ---------------------------------------------------------------------------
# 2c -- Union of polynomial sub-level sets
# ---------------------------------------------------------------------------

def union_of_sets(region_polynomials):
    """
    A union of sets, each given as a list of polynomial inequalities
    (the inner list is the AND-conjunction inside one region).

    Parameters
    ----------
    region_polynomials : sequence of sequences of sympy expressions.
        ``region_polynomials[j]`` is the inequality list for region j
        (each entry >= 0 inside region j).

    Returns
    -------
    The same structure, validated.
    """
    if not isinstance(region_polynomials, Sequence):
        raise TypeError("region_polynomials must be a sequence")
    out = []
    for j, region in enumerate(region_polynomials):
        if not isinstance(region, Sequence):
            raise TypeError(f"region {j} must be a sequence of sympy expressions")
        out.append([sp.sympify(g) for g in region])
    return out


# ---------------------------------------------------------------------------
# Convenience: convert axis-aligned box to polytope inequality list. This
# bridges the existing ``generate_polynomial`` convention with the v2
# polytope API, so a single SOS pipeline can consume both.
# ---------------------------------------------------------------------------

def box_to_polytope(x, L, U):
    """
    Box ``[L_i, U_i]`` -> polytope ``{x : x_i >= L_i and x_i <= U_i}``
    in the inequality form ``[x_i - L_i, U_i - x_i]``.
    """
    n = len(x)
    L = np.asarray(L)
    U = np.asarray(U)
    if L.shape != (n,) or U.shape != (n,):
        raise ValueError("L and U must have length n")
    out = []
    for i in range(n):
        out.append(x[i] - sp.sympify(L[i]))
        out.append(sp.sympify(U[i]) - x[i])
    return out
