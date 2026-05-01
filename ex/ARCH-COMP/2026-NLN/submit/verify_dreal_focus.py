"""
Run dReal (delta-complete SMT) on the benchmarks where Z3 was
problematic: LALO20 (Z3 nlsat timed out on Lie) and SPRE22 / LOVO21
(Z3 found violations -- dReal is faster and may agree, disagree, or
add precision).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)

import sympy as sp

from src.functions.verify_dreal import verify_barrier_dreal


LABELS = ['LALO20_W001', 'LALO20_W005', 'LALO20_W01',
          'LOVO21', 'SPRE22']


def main():
    results_dir = os.path.join(HERE, 'results')
    for label in LABELS:
        cfg_path = os.path.join(results_dir, f'{label}.config.json')
        res_path = os.path.join(results_dir, f'{label}.result.json')
        if not (os.path.isfile(cfg_path) and os.path.isfile(res_path)):
            print(f'  {label}: side-files missing, skipping')
            continue
        cfg = json.load(open(cfg_path))
        res = json.load(open(res_path))
        if 'barrier' not in res:
            print(f'  {label}: no barrier (NOT FOUND or TIMEOUT)')
            continue
        x_syms = sp.symbols(cfg['x_syms'])
        if isinstance(x_syms, sp.Symbol):
            x_syms = (x_syms,)
        p_syms = sp.symbols(cfg.get('p_syms', [])) if cfg.get('p_syms') else ()
        if isinstance(p_syms, sp.Symbol):
            p_syms = (p_syms,)
        sym_table = {str(s): s for s in (list(x_syms) + list(p_syms))}
        dynamics = [sp.sympify(s, locals=sym_table) for s in cfg['dynamics']]
        barrier = sp.sympify(res['barrier'], locals=sym_table)

        pv = cfg.get('p_values', {})
        P_lo = []; P_hi = []
        if p_syms:
            for s in p_syms:
                v = pv.get(str(s), 0.0)
                P_lo.append(v); P_hi.append(v)

        verdict = verify_barrier_dreal(
            barrier=barrier, x_syms=list(x_syms), dynamics=dynamics,
            L_initial=cfg['L_initial'], U_initial=cfg['U_initial'],
            L_unsafe=cfg['L_unsafe'],   U_unsafe=cfg['U_unsafe'],
            L_space=cfg['L_space'],     U_space=cfg['U_space'],
            gamma=res['gamma'], lambda_=res['lambda'],
            p_syms=list(p_syms), P_lo=P_lo, P_hi=P_hi,
            delta=1e-4, timeout_s=120,
            work_dir=os.path.join(results_dir, '_dreal_tmp'),
        )
        out_path = os.path.join(results_dir, f'{label}.dreal.json')
        # Tuple results don't serialise; convert to dict.
        ser = {'overall': verdict.get('overall'), 'delta': verdict.get('delta')}
        if verdict.get('initial'):
            ser['initial'] = verdict['initial'][0]
        if verdict.get('lie'):
            ser['lie'] = verdict['lie'][0]
        ser['unsafe'] = [u[0] for u in verdict.get('unsafe', [])]
        with open(out_path, 'w') as f:
            json.dump(ser, f, indent=2)
        u_str = ', '.join(ser.get('unsafe', [])) or '-'
        print(f'  {label:14s} init={ser.get("initial","-"):8s} '
              f'lie={ser.get("lie","-"):8s} '
              f'unsafe=[{u_str}] -> {ser["overall"]}')


if __name__ == '__main__':
    main()
