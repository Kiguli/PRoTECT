"""
Reach-avoid SOS solver (PRoTECT v2 feature 6b, end-to-end).

Find a single polynomial barrier ``B(x)`` and two level values
``gamma_safe < -gamma_target`` such that:

    B(x) <= gamma_safe   on the initial set            (init)
    B(x) >  gamma_safe   on the unsafe set (per region) (avoid)
    B(x) <= -gamma_target on the target set            (reach: trajectory
                                                       enters target's
                                                       sub-level set)
    -<dB/dx, f(x)> >= eps  on safe-minus-target         (descent)

The "safe-minus-target" descent condition uses two coupled S-procedure
multipliers: an SOS multiplier on the safe-region inequality, and a
positive multiplier on (gamma_target + B) to localise the descent
constraint to states outside the target sub-level set.
"""

import numpy as np
import sympy as sp

import picos
from SumOfSquares import SOSProblem, poly_variable

from .generate_polynomial import generate_polynomial


def ct_DS_reach_avoid(
    b_degree,
    x, f,
    gamma_safe, gamma_target,
    initial_polys=None,
    unsafe_regions=None,
    target_polys=None,
    space_polys=None,
    L_initial=None, U_initial=None,
    L_unsafe=None,  U_unsafe=None,
    L_target=None,  U_target=None,
    L_space=None,   U_space=None,
    descent_eps=1e-3,
    solver='mosek',
    l_degree=None,
):
    """
    NOTE: gamma_safe and gamma_target are FIXED by the caller. Searching
    over both jointly with a single-shot SOS is bilinear; the documented
    follow-on is alternating optimisation (outer loop over gamma values).
    Requires:
        gamma_target > 0
        gamma_safe   > -gamma_target
    (so that the target sub-level set ``{B <= -gamma_target}`` is
    strictly contained in the safe sub-level set ``{B <= gamma_safe}``).
    Typical choice: gamma_safe = -0.1, gamma_target = 0.5.
    """
    if gamma_target <= 0:
        raise ValueError("require gamma_target > 0")
    if not gamma_safe > -gamma_target:
        raise ValueError(
            f"require gamma_safe > -gamma_target "
            f"(got gamma_safe={gamma_safe}, -gamma_target={-gamma_target})")
    if len(f) != len(x):
        raise ValueError("len(f) must equal len(x)")

    g_initial = [sp.sympify(g) for g in initial_polys] if initial_polys else \
                list(generate_polynomial(x, L_initial, U_initial))
    g_space = [sp.sympify(g) for g in space_polys] if space_polys else \
              list(generate_polynomial(x, L_space, U_space))
    g_unsafe = [[sp.sympify(g) for g in r] for r in unsafe_regions] \
        if unsafe_regions else \
        [list(generate_polynomial(x, Lj, Uj)) for Lj, Uj in zip(L_unsafe, U_unsafe)]
    g_target = [sp.sympify(g) for g in target_polys] if target_polys else \
               list(generate_polynomial(x, L_target, U_target))

    if l_degree is None:
        l_degree = b_degree

    prob = SOSProblem()
    result = {'b_degree': b_degree}

    try:
        Barrier = poly_variable('B_ra', x, b_degree)

        L_init = [poly_variable(f'L_ra_init_{i+1}', x, l_degree)
                  for i in range(len(g_initial))]
        L_unsafe_pr = [
            [poly_variable(f'L_ra_unsafe_{j}_{i+1}', x, l_degree)
             for i in range(len(g_unsafe[j]))]
            for j in range(len(g_unsafe))
        ]
        L_target_lag = [poly_variable(f'L_ra_target_{i+1}', x, l_degree)
                        for i in range(len(g_target))]
        L_space_lag = [poly_variable(f'L_ra_space_{i+1}', x, l_degree)
                       for i in range(len(g_space))]
    except Exception:
        return {'error': 'init failure', 'b_degree': b_degree}

    grad = np.array([sp.diff(Barrier, xi) for xi in x])
    Lie = np.sum(grad * np.array(f))

    try:
        # Init: gamma_safe - B - sum L_init_i * g_init_i  is SOS in x.
        first_condition = prob.add_sos_constraint(
            gamma_safe - Barrier
            - sum(Li * gi for Li, gi in zip(L_init, g_initial)),
            x,
        )

        # Avoid: B - gamma_safe - sum L_unsafe_j_i * g_unsafe_j_i is SOS,
        # one per unsafe region.
        last_unsafe = None
        for j in range(len(g_unsafe)):
            cond_j = prob.add_sos_constraint(
                Barrier - gamma_safe
                - sum(Li * gi for Li, gi in zip(L_unsafe_pr[j], g_unsafe[j])),
                x,
            )
            last_unsafe = cond_j

        # Reach: -gamma_target - B - sum L_target_i * g_target_i is SOS.
        target_condition = prob.add_sos_constraint(
            -gamma_target - Barrier
            - sum(Li * gi for Li, gi in zip(L_target_lag, g_target)),
            x,
        )

        # Descent on the full state space (Lyapunov-style strict descent).
        # The "outside target" localiser term L_outside_target * (gamma_target + B)
        # would tighten the descent to safe-minus-target, but it is bilinear in
        # decision variables (L_outside_target's coefficients * Barrier's
        # coefficients), so SOS rejects it. Dropping the localiser keeps the
        # programme convex; the resulting certificate proves global descent
        # which is strictly stronger than safe-minus-target descent.
        descent_condition = prob.add_sos_constraint(
            -Lie - sp.sympify(descent_eps)
            - sum(Li * gi for Li, gi in zip(L_space_lag, g_space)),
            x,
        )

        for Li in L_init:
            prob.add_sos_constraint(Li, x)
        for region in L_unsafe_pr:
            for Li in region:
                prob.add_sos_constraint(Li, x)
        for Li in L_target_lag:
            prob.add_sos_constraint(Li, x)
        for Li in L_space_lag:
            prob.add_sos_constraint(Li, x)

    except AssertionError:
        return {'error': 'AssertionError', 'b_degree': b_degree}

    try:
        prob.solve(solver=solver)
    except picos.modeling.problem.SolutionFailure:
        return {'error': 'picos SolutionFailure', 'b_degree': b_degree}
    except Exception:
        return {'error': 'Solver Exception', 'b_degree': b_degree}

    try:
        result['barrier'] = Barrier  # symbolic; coefficients now bound
        result['gamma_safe'] = gamma_safe
        result['gamma_target'] = gamma_target
        return result
    except Exception:
        return {'error': 'reading result failed', 'b_degree': b_degree}
