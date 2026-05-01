"""
Reach-avoid certificates (PRoTECT v2 feature 6b).

A reach-avoid certificate proves that trajectories from the initial set
eventually enter a target set while never leaving a safe region. The
standard formulation uses a single barrier B(x) with two level sets:
    {B(x) <= -gamma_T} : target (must be reached)
    {B(x) <=  gamma_S} : safe   (must not be left)

with -gamma_T > gamma_S. The Lie derivative is required to be strictly
negative outside the target so that B(x) decreases monotonically until
the target is reached.

SOS conditions:
    B(x) <=  gamma_S      on the initial set
    B(x) >  gamma_S       on the unsafe set                  (avoid)
    B(x) <= -gamma_T      on the target set                  (reach)
    -<dB/dx, f(x)> >= eps on the safe region minus target    (descent)

This module exposes the polynomial-inequality assembly; full SOS solver
integration uses the same scaffolding as ct_DS and is left as a deeper
follow-on (the descent SOS condition needs careful Lagrangian handling
to exclude the target set, typically via two coupled Positivstellensatz
multipliers).
"""

import numpy as np
import sympy as sp


def reach_avoid_conditions(B, x, f, gamma_safe, gamma_target,
                           initial_set_polys, unsafe_set_polys,
                           target_set_polys, descent_eps=1e-3):
    """
    Build the four polynomial inequalities (un-Lagrangianised) for a
    reach-avoid certificate. The caller wraps each with appropriate SOS
    multipliers in their solver.

    Returns
    -------
    dict with keys:
        'initial'  : sympy expr representing  gamma_safe - B
        'avoid'    : sympy expr representing  B - gamma_safe
        'reach'    : sympy expr representing  -gamma_target - B
        'descent'  : sympy expr representing  -<dB/dx, f> - descent_eps

    along with the side conditions ``initial_set_polys``,
    ``unsafe_set_polys``, ``target_set_polys`` (the caller adds them as
    Positivstellensatz multipliers).
    """
    grad = np.array([sp.diff(B, xi) for xi in x])
    descent = -np.sum(grad * f) - sp.sympify(descent_eps)
    return {
        'initial': sp.sympify(gamma_safe) - sp.sympify(B),
        'avoid':   sp.sympify(B) - sp.sympify(gamma_safe),
        'reach':   -sp.sympify(gamma_target) - sp.sympify(B),
        'descent': descent,
        'side': {
            'initial_set_polys': list(initial_set_polys),
            'unsafe_set_polys':  list(unsafe_set_polys),
            'target_set_polys':  list(target_set_polys),
        },
    }
