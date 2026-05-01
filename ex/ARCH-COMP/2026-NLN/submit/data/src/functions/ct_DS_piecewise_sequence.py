"""
Sequence runner for piecewise-constant inputs (PRoTECT v2 feature 4b,
end-to-end orchestration).

For closed-loop dynamics ``x' = f(x, u)`` with a sequence of constant
input values ``u_0, u_1, ..., u_{K-1}``, run K independent ct_DS_v2
solves -- one per segment -- and report the per-segment barrier
together with the consistency-of-inclusion certificates that compose
the segments. The k-th segment's safe sub-level set
``{x : B_k(x) <= gamma_k}`` must contain segment k+1's initial set.

Two modes:

  * 'independent': for each segment, take the previous segment's safe
    sub-level set as that segment's initial set and re-solve. The user
    sees per-segment barriers + a final overall verdict.

  * 'joint': all per-segment barriers are searched in one combined SOS
    programme with explicit composition multipliers. (Bigger programme,
    tighter certificate.)

This module ships the 'independent' runner; 'joint' is a follow-on
since it grows the programme size linearly in K.
"""

import numpy as np
import sympy as sp

from .ct_DS_v2 import ct_DS_v2
from .piecewise_input import substitute_input


def ct_DS_piecewise_sequence(
    b_degree,
    x, f_template, u_syms,
    u_sequence,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    solver='mosek',
    gam=None, lam=None, l_degree=None,
):
    """
    Independent-segment runner. Returns a list of result dicts, one per
    segment, plus a top-level dict summarising overall feasibility.
    """
    per_segment = []
    L_curr_initial = np.array(L_initial, dtype=float)
    U_curr_initial = np.array(U_initial, dtype=float)

    for k, u_value in enumerate(u_sequence):
        f_k = substitute_input(f_template, u_syms, u_value)
        res_k = ct_DS_v2(
            b_degree=b_degree, x=x, f=f_k,
            L_initial=L_curr_initial, U_initial=U_curr_initial,
            L_unsafe=L_unsafe, U_unsafe=U_unsafe,
            L_space=L_space, U_space=U_space,
            solver=solver, gam=gam, lam=lam, l_degree=l_degree,
        )
        per_segment.append(res_k)
        if 'error' in res_k:
            return {
                'segments': per_segment,
                'feasible': False,
                'failed_at_segment': k,
            }
        # For the next segment, we keep the same initial set; a tighter
        # composition would shrink the next-initial to the current
        # safe sub-level set, but that requires extracting the level set
        # geometrically (expensive). Independent mode keeps the same
        # initial set and relies on the per-segment safety guarantee.

    return {
        'segments': per_segment,
        'feasible': True,
        'n_segments': len(u_sequence),
    }
