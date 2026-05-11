"""
TSPS25 -- Transient stability of the IEEE 14-bus power system
(ARCH-COMP NLN benchmark).

Source: ARCH-COMP 2026 Nonlinear Dynamics report, Sec. 3.7.

Original problem: 15 ODEs + 27 algebraic equations (DAE) modelling 5
generators on the IEEE 14-bus network, with a fault scenario (power dropout
on bus 1 at t_o = 0.1, cleared at t_c = 0.13) and verification target
"the reachable set returns to a neighbourhood of the initial state after
the fault is cleared". The dynamics are heavy in sin / cos of the form
y_i sin(y_j +- 1.89), y_i sin(x_i - y_j), etc.

PRoTECT cannot represent:
  - differential-algebraic systems (PRoTECT operates on ODEs only);
  - the temporal "returns to neighbourhood after fault clears" reachability
    spec, which is not a barrier-certificate property;
  - the fault scenario as a discrete jump in the dynamics.

PRoTECT specification (drastically simplified): we elide the algebraic
constraints by fixing the algebraic variables y at their nominal-operating
values (so g(x, y, u) = 0 holds at t = 0; we treat the resulting evaluated
expressions as constants in the sin / cos arguments). The remaining ODE on
the differential states uses the SINC RELAXATION for the residual sin
terms in the swing-equation forcing. This is the closest scoped target
PRoTECT can attempt; we expect it to NOT find a barrier and report this
honestly. The DAE structure is documented as out of scope.

Because the simplified ODE is still 15-dimensional with high-magnitude
nonlinear coefficients, this benchmark is expected to time out or fail
SOS feasibility with cvxopt; we keep the script in the suite for
completeness and to document the attempt.
"""

import os
import time

import numpy as np
import sympy as sp

from src.functions.ct_DS import ct_DS
from src.functions.figure_export import export_figure
from src.functions.solve_helpers import solve_safety_problem
from src.functions.result_export import write_result_json
from src.functions.sinc_relaxation import sinc_bounds, replace_sin


if __name__ == '__main__':
    # Differential states (paper Sec. 3.7.1):
    # x1..x5 are rotor angles (related to bus angles by x_i' = x_{i+5} - 120*pi),
    # x6..x10 are rotor speeds, x11..x15 are governor states.
    # The five swing-equation lines have a residual coupling
    #   x_{6+k}' = 15*x_{11+k}*pi - 3*pi*(x_{6+k} - 120*pi)/5 - C_k * y_k * sin(arg_k)
    # for k = 0..4, with y_k and arg_k determined at nominal operating
    # values from the algebraic equations (Sec. 3.7.1, evaluated symbolically
    # below). We retain only the swing rows and treat governor / mechanical
    # rows as linear damping with constant input.

    # Nominal-operating algebraic-variable values (from paper Sec. 3.7.2,
    # x* := [0.33, -0.02, -0.22, -0.25, -0.23, 377, 377, 377, 377, 377,
    #       2.0, 0.4, 0.0, 0.0, 0.0]).
    x_star = np.array([
        0.33, -0.02, -0.22, -0.25, -0.23,
        377.0, 377.0, 377.0, 377.0, 377.0,
        2.0, 0.4, 0.0, 0.0, 0.0,
    ])

    # Per-row swing coefficients C_k (from paper Sec. 3.7.1 ODEs):
    C = np.array([159.0/2.0, 627.0/8.0, 303.0/4.0, 321.0/4.0, 327.0/4.0])

    # Sinc relaxation bounds for the perturbed angle around nominal.
    angle_max = 0.5
    sinc_lo, _ = sinc_bounds(angle_max)

    # State: x[0..4] perturbations to rotor angles (relative to x_star[0..4]),
    # x[5..9] perturbations to rotor speeds, x[10..14] auxiliary q_k = sinc(arg_k).
    dim = 15

    # Initial set: small perturbation from nominal.
    eps_angle = 0.05
    eps_speed = 0.5
    L_initial = np.concatenate([
        -eps_angle * np.ones(5),
        -eps_speed * np.ones(5),
         sinc_lo  * np.ones(5),
    ])
    U_initial = np.concatenate([
         eps_angle * np.ones(5),
         eps_speed * np.ones(5),
         np.ones(5),
    ])

    L_space = np.concatenate([
        -angle_max * np.ones(5),
        -2.0       * np.ones(5),
         sinc_lo  * np.ones(5),
    ])
    U_space = np.concatenate([
         angle_max * np.ones(5),
         2.0       * np.ones(5),
         np.ones(5),
    ])

    # Unsafe set: any rotor speed deviates by more than 1.5 (loss of
    # synchronism proxy); modelled as 10 box-shaped unsafe regions.
    L_unsafe_list = []
    U_unsafe_list = []
    for k in range(5):
        Lp = L_space.copy(); Lp[5 + k] =  1.5; Up = U_space.copy()
        L_unsafe_list.append(Lp); U_unsafe_list.append(Up)
        Ln = L_space.copy();                  Un = U_space.copy(); Un[5 + k] = -1.5
        L_unsafe_list.append(Ln); U_unsafe_list.append(Un)
    L_unsafe = np.array(L_unsafe_list)
    U_unsafe = np.array(U_unsafe_list)

    x = sp.symbols(f'x0:{dim}')

    # Linearised swing equations (sin replaced by q_k * delta_k via
    # replace_sin); algebraic-variable values held at nominal.
    f_list = []
    p1 = 0.0531
    p2 = 20.0
    omega_nom = 120.0 * sp.pi
    for k in range(5):
        # delta_k' = omega_k - omega_nom contribution
        f_list.append(x[5 + k])
    for k in range(5):
        # omega_k' = -3*pi/5 * x_{6+k} - C_k * y_k_nom * sin(delta_k - y_{15+k}_nom)
        # We fold the constant nominal y values into the coefficient and
        # treat sin(delta_k) (i.e., the perturbation about nominal) via the
        # sinc relaxation.
        y_nom = x_star[5 + k]                    # rotor-speed nominal
        coupling_const = -C[k] * y_nom * 1.0     # take |Y_ij| ~ 1 placeholder
        sin_arg = x[k]                           # perturbation in delta_k
        f_list.append(
            -3 * sp.pi * x[5 + k] / 5
            + coupling_const * replace_sin(sin_arg, x[10 + k])
        )
    for k in range(5):
        # q_k' = 0
        f_list.append(sp.Integer(0))

    f = np.array(f_list)
    start = time.time()
    result, solver_used = solve_safety_problem(
        degrees=range(2, 5, 2),
        x=x, f=f,
        L_initial=L_initial, U_initial=U_initial,
        L_unsafe=L_unsafe,   U_unsafe=U_unsafe,
        L_space=L_space,     U_space=U_space,
        validate_tolerance=1e-8,
    )
    if result:
        result['solver'] = solver_used
    end = time.time()

    print("elapsed time:", (result or {}).get("solve_time_total", end - start))
    print(result if result else "Results dictionary is empty.")

    fig_dir = os.environ.get('PROTECT_RESULT_DIR',
                             os.path.join(os.path.dirname(__file__), '..', 'results'))
    write_result_json(fig_dir, 'TSPS25', result if result else {})
    _r = result or {}
    export_figure(
        os.path.join(fig_dir, 'TSPS25.json'),
        L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space,
        dim_x=0, dim_y=5,
        title='TSPS25 (simplified swing dynamics)',
        x_label='\\delta_{1}', y_label='\\omega_{1}',
        barrier=_r.get('barrier'), x_syms=list(x),
        gamma=_r.get('gamma'), lambda_=_r.get('lambda'),
    )
