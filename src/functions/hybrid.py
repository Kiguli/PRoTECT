"""
Hybrid systems support (PRoTECT v2 features 3a + 3b).

A hybrid automaton ``H = (Q, X, Init, F, Inv, E, G, R)`` has:
    Q   : set of discrete locations (modes)
    X   : continuous state space
    Init: initial set (per-location subsets of X)
    F   : continuous dynamics in each mode, F(q, x)
    Inv : invariant in each mode, x in Inv(q)
    E   : edges (q -> q')
    G   : guards on edges, x in G(e)
    R   : resets on edges, x' = R(e, x)

A hybrid barrier certificate is a tuple ``{B_q : q in Q}`` of per-mode
polynomials satisfying:
    (i)   B_q(x) <= 0 on Init(q) intersected with Inv(q).
    (ii)  B_q(x) >  0 on the unsafe set in mode q intersected with Inv(q).
    (iii) Lie derivative <dB_q/dx, F(q, x)> <= 0 on Inv(q).
    (iv)  Reset-map invariance: for each edge (q -> q', G, R),
              x in G and B_q(x) <= 0 implies B_{q'}(R(x)) <= 0.

The SOS encoding is one ``ct_DS``-style block per mode plus one
inclusion-style block per transition. v2 ships the API and the
inclusion-polynomial assembly; the full multi-mode SOS solve is a
larger integration step.

Crossing-count bookkeeping (3b)
-------------------------------
For benchmarks like LOVO25, an integer auxiliary state ``n_cross`` is
introduced with reset ``n_cross := n_cross + 1`` on the relevant
transitions and bounded by an SOS-encoded box constraint
``0 <= n_cross <= K_max`` plus a parity multiplier
``n_cross * (n_cross - 2) * ... * (n_cross - 2*K_max)`` to enforce
even-only counts (encoding "no odd crossing").
"""

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Mode / edge data classes
# ---------------------------------------------------------------------------

class Mode:
    def __init__(self, name, x, dynamics, invariant_polys,
                 initial_polys=None, unsafe_polys_list=None):
        self.name = name
        self.x = list(x)
        self.dynamics = list(dynamics)
        self.invariant_polys = list(invariant_polys)
        self.initial_polys = list(initial_polys) if initial_polys else []
        self.unsafe_polys_list = list(unsafe_polys_list) if unsafe_polys_list else []


class Edge:
    def __init__(self, src, dst, guard_polys, reset_map):
        self.src = src
        self.dst = dst
        self.guard_polys = list(guard_polys)
        self.reset_map = reset_map  # callable: list of sympy exprs given source state


# ---------------------------------------------------------------------------
# Per-mode SOS conditions (call ct_DS-style block per mode)
# ---------------------------------------------------------------------------

def per_mode_lie_derivative(B, mode):
    grad = np.array([sp.diff(B, xi) for xi in mode.x])
    return np.sum(grad * np.array(mode.dynamics))


# ---------------------------------------------------------------------------
# Reset-map inclusion polynomial (transition condition iv)
# ---------------------------------------------------------------------------

def reset_inclusion_expression(B_src, B_dst, edge):
    """
    Return the polynomial expression
        - B_dst(R(x))      with R the reset map
    so that the SOS condition becomes
        - B_dst(R(x)) - sum lambda_g(x) * g_guard(x) - lambda_src * (- B_src(x))   is SOS.

    The caller wraps with Lagrangians as usual.
    """
    reset_state = edge.reset_map(edge.src.x)
    subs = dict(zip(edge.dst.x, reset_state))
    B_dst_at_reset = sp.sympify(B_dst).subs(subs)
    return -B_dst_at_reset


# ---------------------------------------------------------------------------
# Crossing-count auxiliary state (3b)
# ---------------------------------------------------------------------------

def crossing_count_invariant(n_cross_sym, k_max):
    """
    Polynomial inequalities encoding `0 <= n_cross <= 2*k_max`:
        [n_cross, 2*k_max - n_cross]

    For even-only counts, ADD the equality
        n_cross * (n_cross - 2) * (n_cross - 4) * ... * (n_cross - 2*k_max) == 0
    via the relaxation registry's equality-multiplier route.
    """
    return [n_cross_sym, 2 * sp.sympify(k_max) - n_cross_sym]


def even_count_equality(n_cross_sym, k_max):
    expr = sp.Integer(1)
    for k in range(0, k_max + 1):
        expr = expr * (n_cross_sym - 2 * k)
    return expr
