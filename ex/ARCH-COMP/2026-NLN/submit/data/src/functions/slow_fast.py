"""
Slow-fast / quasi-steady-state model reduction (PRoTECT v2 feature 8b).

Stiff systems (e.g. ROBE25 instances 2 and 3 with gamma in {1e5, 1e7})
have time scales separated by many orders of magnitude. The fast
variables relax to a quasi-steady-state manifold ``M(x_slow)`` defined
by setting their derivatives to zero:

    f_fast(x_slow, x_fast) = 0   ==>   x_fast = phi(x_slow)

Substituting back into ``f_slow`` gives a reduced ODE on the slow
variables alone, which is much better conditioned for SOS.

This module provides the structural helpers; identifying which
variables are "fast" requires either the user's domain knowledge or a
time-scale analysis pass (e.g. via the eigenvalues of df_fast/dx_fast).
"""

import sympy as sp


def quasi_steady_substitution(f_fast, x_slow_syms, x_fast_syms, u_syms=()):
    """
    Solve ``f_fast = 0`` for ``x_fast_syms``. Returns the dict mapping
    each fast variable to a sympy expression in the slow variables.
    Returns None if sympy cannot find a closed form.
    """
    sol = sp.solve(list(f_fast), list(x_fast_syms), dict=True)
    if not sol:
        return None
    chosen = sol[0]
    return {y: chosen[y] for y in x_fast_syms if y in chosen}


def reduce_dynamics(f_slow, slow_to_fast_subs):
    """
    Apply the slow-to-fast substitution dict to the slow dynamics.
    """
    return [sp.sympify(fi).subs(slow_to_fast_subs) for fi in f_slow]
