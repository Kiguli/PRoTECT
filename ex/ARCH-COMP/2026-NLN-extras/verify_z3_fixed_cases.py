"""
Z3 post-hoc verification of the barrier certificates that pass the
PRoTECT v2 pointwise validator as 'clean' or 'warn':

  LALO20/W001, W005, W01     (clean)
  LOVO25                     (warn)
  CVDP23_finite_time         (warn, paper spec b in [1,3], t in [0,7])
  CVDP23_finite_time_fixedB1 (warn, b = 1 fixed,         t in [0,7])

Z3 reasons in exact rational arithmetic (the SOS barrier coefficients
are converted via sympy.Rational.limit_denominator). For each
certificate we check:

  (i)   forall x in X_0:    B(x) <= gamma     (or B(x, 0) <= gamma for FT)
  (ii)  forall x in X_u_j:  B(x) >= lambda    (per unsafe region)
  (iii) forall x in X, p:   <grad_x B, f(x, p)> <= 0   (for time-invariant)
        forall x in X, p, t in [0, T]:  dB/dt + <grad_x B, f> <= 0  (for FT)

Z3 returns unsat (-> condition holds) or sat (-> Z3 found a true
counterexample inside the box). We use a small tolerance (1e-3 by
default) to allow for the SOS solver's floating-point noise; setting
tolerance to 0 reproduces strict checking.
"""

import json
import os
import re
import sys

import numpy as np
import sympy as sp


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RESULTS = os.path.join(HERE, 'results')
sys.path.insert(0, ROOT)

from src.functions.verify_smt import verify_barrier


def _sympify(s):
    s = re.sub(r'(\d)\.e([+-]?\d+)', r'\1.0e\2', s)
    return sp.sympify(s)


def _print_result(label, out):
    def _fmt(triple, name):
        status, cex, viol, t = triple
        verdict = 'OK ' if status == 'unsat' else ('CEX' if status == 'sat' else 'TO?')
        if isinstance(viol, (int, float)) and viol is not None:
            return f'   {name:10s}  {status:8s}  {verdict}  violation={viol:+.3e}  ({t:.1f}s)'
        return f'   {name:10s}  {status:8s}  {verdict}  ({t:.1f}s)'

    print(f'  {label}: overall = {out.get("overall")}, tol = {out.get("tolerance")}')
    print(_fmt(out['initial'], 'init'))
    for j, u in enumerate(out['unsafe']):
        print(_fmt(u, f'unsafe[{j}]'))
    print(_fmt(out['lie'], 'lie'))


def verify_lalo20(inst):
    label = f'LALO20_{inst}'
    rj = os.path.join(RESULTS, label + '.result.json')
    if not os.path.isfile(rj):
        return None
    d = json.load(open(rj))
    barrier = _sympify(d['barrier'])
    gamma = float(d['gamma']); lam = float(d['lambda'])

    inst_map = {'W001': (0.01, 4.5), 'W005': (0.05, 4.5), 'W01': (0.10, 5.0)}
    W, x4_unsafe = inst_map[inst]

    x = sp.symbols('x0:7')
    centre = np.array([1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45])
    L_init = centre - W; U_init = centre + W
    L_space = np.array([0.5, 0.5, 1.0, 1.5, 0.5, 0.05, 0.2])
    U_space = np.array([2.5, 2.5, 4.0, 6.0, 2.5, 0.5,  1.2])
    L_unsafe1 = L_space.copy(); L_unsafe1[3] = x4_unsafe
    f = [1.4*x[2] - 0.9*x[0],
         2.5*x[4] - 1.5*x[1],
         0.6*x[6] - 0.8*x[1]*x[2],
         2 - 1.3*x[2]*x[3],
         0.7*x[0] - x[3]*x[4],
         0.3*x[0] - 3.1*x[5],
         1.8*x[5] - 1.5*x[1]*x[6]]
    print(f'verifying {label} (deg = {sp.Poly(barrier, *x).total_degree()})...')
    return verify_barrier(
        barrier, list(x), f,
        L_init, U_init, [L_unsafe1], [U_space],
        L_space, U_space, gamma, lam,
        tolerance=1e-3, timeout_s=300.0,
    )


def verify_lovo25():
    rj = os.path.join(RESULTS, 'LOVO25.result.json')
    d = json.load(open(rj))
    barrier = _sympify(d['barrier'])
    gamma = float(d['gamma']); lam = float(d['lambda'])
    x0, x1 = sp.symbols('x0 x1')
    L_init = np.array([1.288, 1.0 - 1e-3]); U_init = np.array([1.312, 1.0 + 1e-3])
    L_space = np.array([0.6, 0.6]); U_space = np.array([1.4, 1.4])
    # Unsafe = complement of the box, modeled as four boxes.
    L_unsafe_list = [np.array([0.5, 0.5]), np.array([1.4, 0.5]),
                     np.array([0.5, 0.5]), np.array([0.5, 1.4])]
    U_unsafe_list = [np.array([0.6, 1.5]), np.array([1.5, 1.5]),
                     np.array([1.5, 0.6]), np.array([1.5, 1.5])]
    f = [3*x0 - 3*x0*x1, x0*x1 - x1]
    print('verifying LOVO25...')
    return verify_barrier(
        barrier, [x0, x1], f,
        L_init, U_init, L_unsafe_list, U_unsafe_list,
        L_space, U_space, gamma, lam,
        tolerance=1e-3, timeout_s=60.0,
    )


def verify_cvdp23_ft(label, b_val_for_lie):
    """For finite-time CVDP23: t is treated as an extra state.
    The Lie condition has an extra dB/dt term and the t-box is added
    to the state-space constraints. p_syms is empty (b is fixed for
    the Lie check OR we pass b in p_syms to sweep it)."""
    rj = os.path.join(RESULTS, label + '.result.json')
    if not os.path.isfile(rj):
        return None
    d = json.load(open(rj))
    barrier_full = _sympify(d['barrier'])
    gamma = float(d['gamma']); lam = float(d['lambda'])
    T_horizon = float(d.get('T_horizon', 7.0))
    x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3')
    free_extra = [s for s in barrier_full.free_symbols if str(s) not in {'x0','x1','x2','x3'}]
    t_sym = free_extra[0] if free_extra else None
    # Use a Z3-friendly notation: keep t as an extra "state" variable.
    # We pass b as p_syms when label is the uncertain version. For the
    # fixed-b version we substitute b_val_for_lie directly.
    L_init = np.array([1.25, 2.35, 1.25, 2.35])
    U_init = np.array([1.55, 2.45, 1.55, 2.45])
    L_space = np.array([-3.0]*4); U_space = np.array([3.0]*4)
    L_u1 = L_space.copy(); L_u1[1] = 2.75
    L_u2 = L_space.copy(); L_u2[3] = 2.75
    L_unsafe = [L_u1, L_u2]
    U_unsafe = [U_space.copy(), U_space.copy()]
    # Substitute t = 0 for init; for unsafe we sample at t = T_horizon.
    barrier_init = barrier_full.subs(t_sym, 0) if t_sym is not None else barrier_full
    barrier_T = barrier_full.subs(t_sym, T_horizon) if t_sym is not None else barrier_full

    # For Lie, treat t and (optional) b as additional Z3 variables via p_syms.
    if 'uncertainB' in label or label == 'CVDP23_finite_time':
        b_sym = sp.Symbol('b0')
        b_in_dyn = b_sym
        p_syms_extra = [b_sym]
        P_lo_extra = [1.0]; P_hi_extra = [3.0]
    else:
        b_in_dyn = b_val_for_lie
        p_syms_extra = []
        P_lo_extra = []; P_hi_extra = []
    if t_sym is not None:
        p_syms_extra.append(t_sym)
        P_lo_extra.append(0.0); P_hi_extra.append(T_horizon)

    mu = 1.0
    dyn = [
        x1,
        mu*(1 - x0**2)*x1 + b_in_dyn*(x2 - x0) - x0,
        x3,
        mu*(1 - x2**2)*x3 - b_in_dyn*(x2 - x0) - x2,
    ]
    # Lie for the finite-time barrier: dB/dt + sum dB/dxi * f_i.
    if t_sym is not None:
        dBdt = sp.diff(barrier_full, t_sym)
    else:
        dBdt = sp.Integer(0)
    # The dynamics passed to verify_barrier are the state dynamics only;
    # the dB/dt is folded by augmenting state. The verify_barrier function
    # constructs <grad B, f>, but here we want dB/dt + <grad B, f>. So we
    # pass a synthetic dynamics vector where one of the state symbols is t
    # with dt/dt = 1. We augment x_syms with t, dyn with 1.
    if t_sym is not None:
        x_syms = [x0, x1, x2, x3, t_sym]
        dyn_full = dyn + [sp.Integer(1)]
        L_init_aug = np.append(L_init, 0.0); U_init_aug = np.append(U_init, 0.0)
        L_space_aug = np.append(L_space, 0.0); U_space_aug = np.append(U_space, T_horizon)
        # For the init check, t=0 box: tightened so t exactly 0.
        L_unsafe_aug = []; U_unsafe_aug = []
        for L_u, U_u in zip(L_unsafe, U_unsafe):
            L_unsafe_aug.append(np.append(L_u, 0.0))
            U_unsafe_aug.append(np.append(U_u, T_horizon))
    else:
        x_syms = [x0, x1, x2, x3]
        dyn_full = dyn
        L_init_aug = L_init; U_init_aug = U_init
        L_space_aug = L_space; U_space_aug = U_space
        L_unsafe_aug = L_unsafe; U_unsafe_aug = U_unsafe

    # If b is in p_syms_extra, keep p_syms otherwise empty.
    if 'uncertainB' in label or label == 'CVDP23_finite_time':
        p_syms_ret = [sp.Symbol('b0')]
        P_lo_ret = [1.0]; P_hi_ret = [3.0]
    else:
        p_syms_ret = []; P_lo_ret = []; P_hi_ret = []

    print(f'verifying {label} (finite-time, deg = ?)...')
    return verify_barrier(
        barrier_full, x_syms, dyn_full,
        L_init_aug, U_init_aug, L_unsafe_aug, U_unsafe_aug,
        L_space_aug, U_space_aug, gamma, lam,
        p_syms=p_syms_ret, P_lo=P_lo_ret, P_hi=P_hi_ret,
        tolerance=1e-3, timeout_s=120.0,
    )


def main():
    results = {}
    for inst in ['W001', 'W005', 'W01']:
        out = verify_lalo20(inst)
        if out is not None:
            results[f'LALO20_{inst}'] = out
            _print_result(f'LALO20_{inst}', out)
            print()
    out = verify_lovo25()
    if out is not None:
        results['LOVO25'] = out
        _print_result('LOVO25', out)
        print()
    for label, b_val in [('CVDP23_finite_time', 2.0),
                         ('CVDP23_finite_time_fixedB1_d2_k2', 1.0)]:
        out = verify_cvdp23_ft(label, b_val)
        if out is not None:
            results[label] = out
            _print_result(label, out)
            print()
    # Save summary.
    summary = {}
    for k, v in results.items():
        summary[k] = {
            'overall': v.get('overall'),
            'initial': list(v['initial'][:2]) + [str(v['initial'][2]), v['initial'][3]],
            'unsafe':  [list(u[:2]) + [str(u[2]), u[3]] for u in v['unsafe']],
            'lie':     list(v['lie'][:2]) + [str(v['lie'][2]), v['lie'][3]],
            'tolerance': v.get('tolerance'),
        }
    out_path = os.path.join(RESULTS, 'z3_verification_summary.json')
    with open(out_path, 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)
    print(f'\nZ3 summary written to {out_path}')


if __name__ == '__main__':
    main()
