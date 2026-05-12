"""
Test the per-condition strict-positivity margins (init_margin,
unsafe_margin, lie_margin) introduced in ct_DS_robust on PRoTECT v2.

Theory: with delta > 0, each SOS expression -B - sum L_i g_i + gamma
is forced to be SOS GREATER than delta. The certificate then has a
RIGOROUS pointwise margin of at least delta on the asserted set (modulo
solver tolerance ~1e-8 times polynomial basis amplification), so the
pointwise validator should agree.

We sweep delta values on:
  - LALO20/{W001, W005, W01}: known-clean with HUGE negative slacks
    (-3 to -17), should tolerate large delta
  - The infinite-time CVDP23 / CVDP22 / LOVO21 / ROBE25 / TRAF22:
    known-failing with positive pointwise slacks ~1e-4 to 1e-1,
    should benefit from delta if certificate has room
"""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import numpy as np
import sympy as sp

from src.functions.ct_DS_robust import ct_DS_robust


def cvdp_dyn(b, x):
    return np.array([x[1], (1 - x[0]**2)*x[1] + b*(x[2]-x[0]) - x[0],
                     x[3], (1 - x[2]**2)*x[3] - b*(x[2]-x[0]) - x[2]])


def run_one(label, b_degree, dim, x, f, L_init, U_init, L_u, U_u, L_sp, U_sp,
            p_syms=(), P_lo=(), P_hi=(), deltas=(0.0, 1e-6, 1e-4, 1e-2)):
    print(f'\n=== {label} ===')
    for delta in deltas:
        t0 = time.time()
        res = ct_DS_robust(
            b_degree=b_degree, dim=dim,
            L_initial=L_init, U_initial=U_init,
            L_unsafe=L_u, U_unsafe=U_u,
            L_space=L_sp, U_space=U_sp,
            x=x, f=f, p_syms=p_syms, P_lo=P_lo, P_hi=P_hi,
            margin=0.0, solver='mosek',
            validate_sos=True, validate_tolerance=1e-8,
            init_margin=delta, unsafe_margin=delta, lie_margin=delta)
        dt = time.time() - t0
        if 'error' in res:
            print(f'  delta={delta:7.0e} FAIL ({res["error"]}) ({dt:.0f}s)')
            continue
        pw = res.get('pointwise', {})
        if 'init_slack' in pw:
            print(f'  delta={delta:7.0e} gamma={res["gamma"]:.3g} lambda={res["lambda"]:.3g}  '
                  f'init={pw["init_slack"]:+.2e} unsafe={pw["unsafe_slack"]:+.2e} '
                  f'lie={pw["lie_slack"]:+.2e}  verdict={pw["verdict"]} ({dt:.0f}s)')


def main():
    # -- LALO20 (known-clean; should tolerate large delta) --------------
    for inst, W, x4u in [('W001', 0.01, 4.5), ('W005', 0.05, 4.5), ('W01', 0.10, 5.0)]:
        x = sp.symbols('x0:7')
        centre = np.array([1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45])
        L_init = centre - W; U_init = centre + W
        L_sp = np.array([0.5, 0.5, 1.0, 1.5, 0.5, 0.05, 0.2])
        U_sp = np.array([2.5, 2.5, 4.0, 6.0, 2.5, 0.5,  1.2])
        L_u1 = L_sp.copy(); L_u1[3] = x4u
        f = [1.4*x[2] - 0.9*x[0], 2.5*x[4] - 1.5*x[1],
             0.6*x[6] - 0.8*x[1]*x[2], 2 - 1.3*x[2]*x[3],
             0.7*x[0] - x[3]*x[4], 0.3*x[0] - 3.1*x[5],
             1.8*x[5] - 1.5*x[1]*x[6]]
        # LALO20 already has slack ~ -3 to -17, so try deltas up to 0.5.
        run_one(f'LALO20_{inst}', 2, 7, x, np.array(f),
                L_init, U_init, [L_u1], [U_sp.copy()], L_sp, U_sp,
                deltas=(0.0, 0.1, 0.5, 1.0))

    # -- CVDP22 (b=70, infinite time): known-failing -----------------
    x = sp.symbols('x0:4')
    f = cvdp_dyn(70.0, x)
    L_init = np.array([1.25, 2.35, 1.25, 2.35])
    U_init = np.array([1.55, 2.45, 1.55, 2.45])
    L_sp = np.array([-3.0, -4.0, -3.0, -4.0]); U_sp = np.array([3.0, 4.0, 3.0, 4.0])
    L_u1 = L_sp.copy(); L_u1[1] = 3.7
    L_u2 = L_sp.copy(); L_u2[3] = 3.7
    run_one('CVDP22 (b=70)', 4, 4, x, f,
            L_init, U_init, [L_u1, L_u2], [U_sp.copy(), U_sp.copy()], L_sp, U_sp,
            deltas=(0.0, 1e-6, 1e-4))

    # -- CVDP23 / b=2 (infinite time): known-failing -----------------
    f = cvdp_dyn(2.0, x)
    L_sp = np.array([-3.0]*4); U_sp = np.array([3.0]*4)
    L_u1 = L_sp.copy(); L_u1[1] = 2.75
    L_u2 = L_sp.copy(); L_u2[3] = 2.75
    run_one('CVDP23 b=2', 4, 4, x, f,
            L_init, U_init, [L_u1, L_u2], [U_sp.copy(), U_sp.copy()], L_sp, U_sp,
            deltas=(0.0, 1e-6, 1e-4))

    # -- LOVO21 (Lorenz, with shrunken envelope = LOVO21_proposed) ---
    x = sp.symbols('x0:3')
    sigma_, rho_, beta_ = 10.0, 28.0, sp.Rational(8, 3)
    f = np.array([sigma_*(x[1]-x[0]),
                  x[0]*(rho_-x[2]) - x[1],
                  x[0]*x[1] - beta_*x[2]])
    L_init = np.array([0.9, -0.01, -0.01]); U_init = np.array([1.1, 0.01, 0.01])
    L_sp = np.array([-15.0, -15.0, 0.0]); U_sp = np.array([15.0, 15.0, 30.0])
    L_u1 = np.array([15.0, L_sp[1], L_sp[2]])
    run_one('LOVO21 (proposed envelope)', 4, 3, x, f,
            L_init, U_init, [L_u1], [U_sp.copy()], L_sp, U_sp,
            deltas=(0.0, 1e-4, 1e-2, 1e-1))


if __name__ == '__main__':
    main()
