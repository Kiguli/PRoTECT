"""
SPRE22 -- Spacecraft rendezvous (ARCH-COMP NLN benchmark).

Source: ARCH-COMP 2026 Nonlinear Dynamics report, Sec. 3.6.

PRoTECT v2 reformulation, with two fixes vs the previous attempt:

  1. State rescaling (positions / 1000): the original state space spans
     1000 m in x and 500 m in y, while velocities span ~5 m/min. That
     200x dynamic range collapsed MOSEK's interior-point conditioning;
     SMT verification on the resulting certificate found an init-side
     violation of ~9.2e3 -- i.e. the SOS solution was numerically
     feasible but mathematically invalid. Working in u = x/1000,
     v = y/1000 (positions in km), with velocities unchanged, brings
     all state magnitudes to O(1).

  2. Explicit lambda-gamma margin: the previous run had gamma = 1.25,
     lambda = 12.65; while strictly lambda > gamma, MOSEK's tolerance
     left no real separation. Forcing lambda >= gamma + 0.1 via
     ct_DS_robust's `margin` parameter gives Z3 a verifiable gap.

State (rescaled):
    x[0] = u  = x / 1000     in [-1.0, 0.05]    (km)
    x[1] = v  = y / 1000     in [-0.5, 0.05]    (km)
    x[2] = vx                in [-10, 10]       (m/min)
    x[3] = vy                in [-10, 10]       (m/min)

Out of scope (documented in the report discussion): hybrid-mode
switching, line-of-sight cone (non-axis-aligned), |v| velocity ball
(quadratic, not box).
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS_robust import ct_DS_robust
from src.functions.figure_export import export_figure
from src.functions.solve_helpers import solve_safety_problem
from src.functions.result_export import write_result_json


# Constants from the SpaceEx SPRE21.xml (paper parameter values).
r   = 42164e3                    # geostationary radius [m]
mu_ = 3.986e14 * 60**2           # gravitational parameter [m^3 / min^2]
m_c = 500.0                       # spacecraft mass [kg]
n   = (mu_ / r**3)**0.5           # mean motion [1/min]
g_lin = mu_ / r**3

# K1 (Approaching mode) LQR gains.
K1 = np.array([
    [-28.8287,   0.1005, -1449.9754,    0.0046],
    [ -0.087,  -33.2562,     0.00462, -1451.5013],
])

if __name__ == '__main__':
    dim = 4

    # Original (un-rescaled) state. Rescaling positions by 1000 made the
    # dynamics too stiff for SOS (position rates O(0.01) vs velocity rates
    # O(100)). The previous run's huge init-side violation came from
    # MOSEK terminating at a near-feasible point with no real margin
    # between gamma and lambda; we fix that by requiring an explicit
    # margin >= 1.0 instead.
    L_initial = np.array([-925.0, -425.0, 0.0, 0.0])
    U_initial = np.array([-875.0, -375.0, 5.0, 5.0])

    L_space = np.array([-1000.0, -500.0, -10.0, -10.0])
    U_space = np.array([   50.0,   50.0,  10.0,  10.0])

    V_max = 5.0
    L_u1 = L_space.copy(); L_u1[2] =  V_max; U_u1 = U_space.copy()
    L_u2 = L_space.copy();                   U_u2 = U_space.copy(); U_u2[2] = -V_max
    L_u3 = L_space.copy(); L_u3[3] =  V_max; U_u3 = U_space.copy()
    L_u4 = L_space.copy();                   U_u4 = U_space.copy(); U_u4[3] = -V_max

    L_unsafe = np.array([L_u1, L_u2, L_u3, L_u4])
    U_unsafe = np.array([U_u1, U_u2, U_u3, U_u4])

    x = sp.symbols(f'x0:{dim}')  # x[0]=x, x[1]=y, x[2]=vx, x[3]=vy

    # Polynomialised gravity (first-order Taylor at origin):
    f1 = x[2]
    f2 = x[3]
    f3 = (n**2 + K1[0, 0]/m_c)*x[0] + (2*n + K1[0, 3]/m_c)*x[3] \
         + K1[0, 1]/m_c * x[1] + K1[0, 2]/m_c * x[2] - g_lin * x[0]
    f4 = (n**2 + K1[1, 1]/m_c)*x[1] + (K1[1, 2]/m_c - 2*n)*x[2] \
         + K1[1, 0]/m_c * x[0] + K1[1, 3]/m_c * x[3] - g_lin * x[1]
    f = np.array([f1, f2, f3, f4])
    start = time.time()
    result, solver_used = solve_safety_problem(
        degrees=range(2, 7, 2),
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        validate_tolerance=1e-8,
    margin=10.0, mosek_tol=1e-10,
    )
    if result:
        result['solver'] = solver_used
    end = time.time()

    print("elapsed time:", (result or {}).get("solve_time_total", end - start))
    print(result if result else "Results dictionary is empty.")

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'SPRE22', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, 'SPRE22.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=0, dim_y=1,
        title='SPRE22 (margin=1.0)',
        x_label='x [m]', y_label='y [m]',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
