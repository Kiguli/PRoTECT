"""
Finite-time reach-tube width bounds (PRoTECT v2 feature 6a).

The standard PRoTECT pipeline certifies time-UNBOUNDED safety: it finds
``B(x)`` such that the unsafe set is unreachable for all t in [0, infty).
Several ARCH-COMP NLN benchmarks instead ask for *finite-time* metrics:

    * ROBE25  : ``width(x + y + z)`` at ``t = 40 s`` must be < 1e-5.
    * CVDP23  : reach-set characterisation at ``t = 7``.
    * TSPS25  : "reach back to neighbourhood after fault clears".

These are not safety properties in the time-unbounded sense; they're
*reachability* questions on a fixed horizon ``[0, T]``. v2 adds a
time-varying barrier ``B(x, t)`` parameterised as

    B(x, t) = B_0(x) + t * B_1(x) + t^2 * B_2(x) + ... + t^d * B_d(x)

and certifies an inclusion of the reach set in a target sub-level set.

SOS conditions (Positivstellensatz):

    B(x_0, 0) <= 0                           on initial set
    B(x, T) >= -epsilon                      on the target sub-level
    -d/dt B(x, t) - <dB/dx, f(x)> >= 0       for t in [0, T] inside state

(Plus the usual Lagrangian multipliers for state space + time interval.)

Implementation: assemble the time-augmented SOS programme. We expose the
basic skeleton; the full integration with PRoTECT's `add_sos_constraint`
pipeline is the same shape as `ct_DS` but with the time variable t added
to every constraint and a t-box S-procedure multiplier (t * (T - t)).
"""

import numpy as np
import sympy as sp


def time_polynomial_barrier(x, t, degree, time_orders=2, name='Btime'):
    """
    Build a symbolic time-varying barrier
        B(x, t) = sum_{k=0}^{time_orders} t^k * B_k(x)
    where each ``B_k`` is a fresh polynomial of total degree ``degree``
    in x (returned along with the symbol list, so the caller can hand
    them to the SOS programme as decision variables).

    Returns
    -------
    B  : sympy expression in x and t
    Bk : list of sympy expressions, the time-coefficient polynomials
    """
    Bk_list = []
    for k in range(time_orders + 1):
        # symbolic coefficient polynomial; the caller turns these into
        # SumOfSquares poly_variable instances.
        Bk = sp.Symbol(f'{name}_t{k}')
        Bk_list.append(Bk)
    B = sum(Bk_list[k] * t**k for k in range(time_orders + 1))
    return B, Bk_list


def time_box_polynomial(t, T):
    """
    Box S-procedure polynomial for the time interval [0, T]:
        g_t(t) = t * (T - t)    >= 0  iff  t in [0, T]
    """
    return t * (sp.sympify(T) - t)


def lie_derivative_time_dep(B, x, t, f):
    """
    Compute dB/dt + <dB/dx, f(x, t)> for a time-varying barrier B and
    dynamics f (which itself may depend on t for time-varying inputs).
    """
    dBdt = sp.diff(B, t)
    grad_x = np.array([sp.diff(B, xi) for xi in x])
    return dBdt + np.sum(grad_x * f)
