"""
Block-structured / compositional barriers (PRoTECT v2 feature 8c).

For weakly-coupled multi-agent or networked systems
    x_i' = f_i(x_i) + sum_{j != i} h_{ij}(x_i, x_j),
search for per-subsystem barriers ``B_i(x_i)`` plus interconnection
"supply rate" certificates that ensure the whole-system safety follows
from a small-gain or dissipativity argument. Each per-subsystem SOS
programme is independent and small.

The certificate looks like
    sum_i alpha_i * Lie B_i + sum_{i, j} S_ij(x_i, x_j) <= 0,
where each ``S_ij`` is a polynomial supply rate certificate that the
edge between subsystem i and j "uses up" no more energy than j's
storage function provides.

Reference: Anand, Lavaei, Soudjani, "Compositional construction of
control barrier functions for networks of stochastic systems".

This module exposes the per-subsystem and interconnection skeletons.
The full compositional SOS solve is a larger integration step.
"""

import numpy as np
import sympy as sp


def per_subsystem_lie(B_i, x_i, f_i):
    grad = np.array([sp.diff(B_i, xi) for xi in x_i])
    return np.sum(grad * np.array(f_i))


def supply_rate_polynomial(x_i, x_j, S_ij_template):
    """
    Caller supplies the supply-rate polynomial ``S_ij`` as a sympy
    expression in ``x_i`` and ``x_j`` (e.g. ``-||x_i - x_j||^2`` for
    consensus-style coupling). Returns the expression ready for use in
    the compositional SOS sum.
    """
    return sp.sympify(S_ij_template)


def small_gain_inequality(per_subsystem_lies, supply_rates):
    """
    The compositional certificate's central inequality:
        sum Lie B_i + sum S_ij <= 0
    Returns the symbolic LHS of the inequality. The caller asserts SOS
    on -(this expression) to certify network-wide safety.
    """
    total = sum(per_subsystem_lies) + sum(supply_rates)
    return total
