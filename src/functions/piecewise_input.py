"""
Piecewise-constant time-varying inputs (PRoTECT v2 feature 4b).

Decompose a closed-loop trajectory ``x' = f(x, u(t))`` with
piecewise-constant control inputs ``u(t) = u_k`` for ``t in [t_k, t_{k+1}]``
into a sequence of fixed-input continuous-time problems. For each segment
we run the standard ct_DS programme with ``u_k`` substituted into the
dynamics; the safety guarantee composes by requiring the level set at
the end of segment k to imply the initial-set condition of segment k+1.

This reduces the problem to ``K`` independent ct_DS programmes plus a
``K-1`` set of consistency conditions of the form:

    {x : B_k(x) <= gamma_k}  subset  {x : B_{k+1}(x) <= gamma_{k+1}}

The consistency conditions are themselves SOS feasibility checks; we
emit them as additional ``add_sos_constraint`` calls.

For the v2 milestone we ship the segment-runner + a helper that emits
the consistency conditions; tying them together in one SOS programme
(joint search over all B_k) is left as a follow-on, since it grows the
programme size linearly in K.
"""

import numpy as np
import sympy as sp


def substitute_input(f, u_syms, u_values):
    """
    Replace each input symbol in u_syms with the corresponding numeric
    value in the dynamics array f.
    """
    if len(u_syms) != len(u_values):
        raise ValueError("u_syms and u_values must match in length")
    subs = dict(zip(u_syms, [sp.sympify(v) for v in u_values]))
    return np.array([sp.sympify(fi).subs(subs) for fi in f])


def segment_problems(f, u_syms, u_sequence):
    """
    Given a parametric dynamics array ``f(x, u)`` and a list of input
    values ``u_sequence = [u_0, u_1, ..., u_{K-1}]``, return a list of
    K dynamics arrays each with the input value substituted.
    """
    return [substitute_input(f, u_syms, u_k) for u_k in u_sequence]


def consistency_inclusion_polynomial(B_k, gamma_k, B_kplus1, gamma_kplus1):
    """
    Return the polynomial inequality whose >= 0 SOS-ness encodes
        {x : B_k(x) <= gamma_k}  subset  {x : B_{k+1}(x) <= gamma_{k+1}}

    Equivalent to: gamma_kplus1 - B_kplus1(x) >= 0 whenever
                  gamma_k       - B_k(x)        >= 0.
    Standard Positivstellensatz template:
        (gamma_kplus1 - B_kplus1) - lambda(x) * (gamma_k - B_k) is SOS,
    with lambda an SOS multiplier. The caller is responsible for adding
    `lambda` as an SOS variable and assembling the constraint in their
    SOS programme; we return the un-multipliered polynomial expressions
    so the caller has full control of the multiplier choice.

    Returns
    -------
    head : sympy expr  -- the LHS (gamma_kplus1 - B_kplus1)
    side : sympy expr  -- the RHS factor (gamma_k - B_k)
    """
    head = sp.sympify(gamma_kplus1) - sp.sympify(B_kplus1)
    side = sp.sympify(gamma_k) - sp.sympify(B_k)
    return head, side
