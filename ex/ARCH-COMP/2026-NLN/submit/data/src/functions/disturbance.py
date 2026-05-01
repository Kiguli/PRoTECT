"""
Bounded-disturbance robust SOS (PRoTECT v2 feature 4a).

Mathematically identical to ``ct_DS_robust`` (feature 7a): a bounded
time-varying disturbance ``w(t) in W`` is treated exactly like an
uncertain-but-bounded parameter, with a fresh box S-procedure multiplier
in the Lie-derivative SOS constraint.

The distinction is conceptual / API-level:
    * a *parameter* is unknown but constant for the trajectory.
    * a *disturbance* may vary arbitrarily in time, but the SOS programme
      enforces robustness over all admissible values at every instant,
      which is strictly stronger.

For the Positivstellensatz encoding, both reduce to the same SOS
programme. This module provides a thin re-export of ``ct_DS_robust``
under the disturbance terminology, so benchmark scripts can be
self-documenting.
"""

from .ct_DS_robust import ct_DS_robust


def ct_DS_disturbed(
    b_degree, dim,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    x, f,
    w_syms, W_lo, W_hi,
    **kwargs,
):
    """
    Wrapper: dynamics ``x' = f(x, w)`` with disturbance ``w(t) in [W_lo, W_hi]``.
    Search for a barrier B(x) robust against every admissible w.
    """
    return ct_DS_robust(
        b_degree=b_degree, dim=dim,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe, U_unsafe=U_unsafe,
        L_space=L_space, U_space=U_space,
        x=x, f=f,
        p_syms=w_syms, P_lo=W_lo, P_hi=W_hi,
        **kwargs,
    )
