"""
Differential-algebraic systems (PRoTECT v2 features 5a + 5b).

For DAEs of the form
    x' = f(x, y, u)
    0  = g(x, y, u)
with ``g`` index-1 (i.e. ``J = dg/dy`` is invertible at the solution),
the implicit-function theorem gives ``y = phi(x, u)`` locally. PRoTECT
v2 supports two strategies:

5a. Index-1 elimination
    If ``g`` is polynomial and one can solve g = 0 explicitly for ``y``
    in terms of ``(x, u)``, substitute back into ``f`` to obtain an ODE
    on ``x`` alone. Run the standard ct_DS pipeline on that ODE.

5b. Manifold-restricted SOS
    Treat the algebraic constraint ``g(x, y, u) = 0`` as a
    Positivstellensatz EQUALITY multiplier. The barrier search is over
    polynomial ``B(x, y)``, and the SOS conditions are augmented with
    ``lambda_g(x, y) * g(x, y, u)`` (un-sign-restricted Lagrangian).
    This is the same pattern as the relaxation-registry equality
    multipliers (sqrt, inv_power); plumbing it through ct_DS is the
    same code change.

This module exposes the elimination helper. The manifold-restricted SOS
piece reuses the same equality-multiplier hook described in
``relaxations.py``.
"""

import sympy as sp


def eliminate_algebraic(g_polys, y_syms, x_syms, u_syms=()):
    """
    Solve ``g_polys = 0`` for ``y_syms`` symbolically. Returns a dict
    mapping each y symbol to a sympy expression in (x, u), or None if
    sympy cannot find a closed form.

    For polynomial g of moderate complexity sympy returns a Groebner
    basis-like result. Use cautiously: the resulting expressions may
    NOT be polynomial (typical: rational, occasionally radicals). The
    caller should run the result through the relaxation registry
    (relax_inv_power / relax_sqrt) to recover SOS-friendly substitutes.
    """
    sol = sp.solve(list(g_polys), list(y_syms), dict=True)
    if not sol:
        return None
    chosen = sol[0]
    # Project out any solutions not in y_syms.
    return {y: chosen[y] for y in y_syms if y in chosen}


def substitute_algebraic(f, y_substitutions):
    """
    Apply the dict ``{y_i: expr_i}`` from ``eliminate_algebraic`` to the
    dynamics array ``f``. Returns a new sympy array with y's replaced.
    """
    return [sp.sympify(fi).subs(y_substitutions) for fi in f]
