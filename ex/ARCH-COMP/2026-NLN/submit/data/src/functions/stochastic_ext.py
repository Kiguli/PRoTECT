"""
Stochastic extensions (PRoTECT v2 feature 9).

PRoTECT already supports Gaussian-noise stochastic systems via
``ct_SS`` and ``dt_SS``. v2 adds three planned extensions:

  9a. Sub-Gaussian / bounded-variance non-Gaussian noise
      Replace the Gaussian-specific Lie expectation
      ``E[Lie B] = grad B . f + 0.5 trace(sigma^T Hess(B) sigma)``
      with a worst-case bound over the noise's moment-generating function
      class. The same SOS scaffold applies; only the Lie expectation
      formula changes.

  9b. Jump-diffusion (compound Poisson)
      Add a generator term
      ``+ lambda_jump * (E[B(x + Delta)] - B(x))``
      where Delta is the jump distribution. The expectation can be
      bounded if Delta has bounded support (relaxation registry).

  9c. Distributionally robust noise
      Worst-case over a moment-bounded ambiguity set; uses the dual
      moment-SOS hierarchy.

This module provides the API and stubs; full SOS-pipeline integration
follows the same shape as `ct_SS` with an alternative expectation
oracle.
"""

import sympy as sp


def expected_lie_subgaussian(B, x, f, sigma, sub_gaussian_proxy_var):
    """
    Worst-case Lie expectation under a sub-Gaussian noise term with
    proxy variance ``sub_gaussian_proxy_var``. Bounds the moment
    generating function as exp(t^2 * proxy_var / 2) and uses the second
    moment for the Hessian term:
        E[Lie B] <= grad B . f + 0.5 * trace(sigma^T Hess(B) sigma) * proxy_var
    """
    grad = [sp.diff(B, xi) for xi in x]
    drift = sum(grad[i] * f[i] for i in range(len(x)))
    hess = sp.Matrix([[sp.diff(B, xi, xj) for xj in x] for xi in x])
    sigma_m = sp.Matrix(sigma)
    cov = sigma_m.T * hess * sigma_m
    diff = sp.Rational(1, 2) * cov.trace() * sp.sympify(sub_gaussian_proxy_var)
    return drift + diff


def jump_diffusion_generator(B, x, f, sigma, jump_rate, jump_subs):
    """
    Skeleton for the jump-diffusion infinitesimal generator. ``jump_subs``
    is a dict mapping each ``x_i`` to ``x_i + Delta_i`` (sympy expr).
    The full expectation E[B(x + Delta)] over the jump distribution must
    be supplied by the caller (here we just emit the symbolic generator
    structure).
    """
    drift_diffusion = expected_lie_subgaussian(B, x, f, sigma, 1)
    jumped = sp.sympify(B).subs(jump_subs)
    jump_term = sp.sympify(jump_rate) * (jumped - sp.sympify(B))
    return drift_diffusion + jump_term
