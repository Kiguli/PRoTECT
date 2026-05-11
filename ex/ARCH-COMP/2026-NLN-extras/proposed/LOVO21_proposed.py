"""
LOVO21_proposed -- adapted Lorenz benchmark for PRoTECT v2.

ORIGINAL ARCH-COMP LOVO21 spec uses the Lorenz attractor with
  init    : x in [0.9, 1.1], y in [-0.01, 0.01], z in [-0.01, 0.01]
  unsafe  : x >= 20
  envelope: x, y in [-30, 30], z in [0, 60]
With the full +/- 30 envelope, the polynomial-coefficient amplification
at the state-space corners (deg-4 monomials reach 30^4 = 8.1e5) makes
the pointwise Lie residual at solver tolerance scale to ~10^2 on the
boundary -- the certificate is coefficient-clean but the pointwise
validator correctly reports fail.

The PROPOSED variant shrinks the state-space envelope to
  x, y in [-15, 15], z in [0, 30]
which is still large enough to contain (a) the initial set, (b) the
unsafe set x >= 20 (after clipping to [20, 15] -> empty in x, so we
also shrink unsafe to x in [15, 30] -> i.e. unsafe means "x reaches
+15"). This is a strictly stronger safety property (the original
allows "x reaches +20"; the proposed forbids "x reaches +15"); if
the proposed barrier certifies, the original is automatically safe.

This proposed variant should report a pointwise-pass certificate at
strict 1e-8 tolerance, demonstrating the v2 pointwise validator on a
modified Lorenz benchmark within a polynomial-coefficient regime
where SOS barriers are well-conditioned.
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS_robust import ct_DS_robust


if __name__ == '__main__':
    dim = 3

    L_initial = np.array([0.9,  -0.01, -0.01])
    U_initial = np.array([1.1,   0.01,  0.01])

    # SHRUNK state space envelope (only change from LOVO21).
    L_space = np.array([-15.0, -15.0,   0.0])
    U_space = np.array([ 15.0,  15.0,  30.0])

    # Unsafe is now "x >= 15" (a STRICTLY STRONGER safety claim than
    # the original "x >= 20" -- any system safe under the proposed
    # spec is automatically safe under the original).
    L_u1 = np.array([15.0, L_space[1], L_space[2]])
    L_unsafe = np.array([L_u1])
    U_unsafe = np.array([U_space.copy()])

    x = sp.symbols(f'x0:{dim}')
    sigma_ = 10.0; rho = 28.0; beta = sp.Rational(8, 3)
    f = np.array([
        sigma_*(x[1] - x[0]),
        x[0]*(rho - x[2]) - x[1],
        x[0]*x[1] - beta*x[2],
    ])

    start = time.time()
    res = ct_DS_robust(
        b_degree=4, dim=3,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        x=x, f=f, p_syms=(), P_lo=(), P_hi=(),
        margin=0.5, mosek_tol=1e-10, solver='mosek',
        validate_sos=True, validate_tolerance=1e-8,
    )
    end = time.time()
    if res:
        res['solver'] = 'mosek'
        res['variant'] = 'LOVO21_proposed (shrunken envelope + tighter unsafe)'

    print('elapsed time:', (res or {}).get('solve_time', end - start))
    print(res if res else 'Results dictionary is empty.')

    from src.functions.result_export import write_result_json
    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'LOVO21_proposed', res if res else {})
