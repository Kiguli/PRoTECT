"""
Run SMT verification on every result + config side-file produced by
the latest local run, and write per-benchmark <label>.verify.json files
plus a summary table.

Reads:
    submit/results/<label>.result.json  (barrier, gamma, lambda)
    submit/results/<label>.config.json  (dynamics, L/U, parameters)

Writes:
    submit/results/<label>.verify.json  (z3 verdicts per condition)
    submit/results/verify_summary.txt   (human-readable table)
"""

import glob
import json
import os
import sys

import sympy as sp


# Make the repo root importable so `src.functions.verify_smt` resolves
# even when running from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.functions.verify_smt import verify_barrier  # noqa: E402


def load_pair(results_dir, label):
    res_path = os.path.join(results_dir, f'{label}.result.json')
    cfg_path = os.path.join(results_dir, f'{label}.config.json')
    if not (os.path.isfile(res_path) and os.path.isfile(cfg_path)):
        return None, None
    with open(res_path) as f:
        res = json.load(f)
    with open(cfg_path) as f:
        cfg = json.load(f)
    return res, cfg


def reify_symbols(names):
    return sp.symbols([str(n) for n in names])


def run_one(label, res, cfg, tolerance=0.0, timeout_s=60.0):
    if 'barrier' not in res:
        return {'label': label, 'overall': 'no-barrier', 'reason': res.get('error', 'no result')}
    x_syms = reify_symbols(cfg['x_syms'])
    p_syms = reify_symbols(cfg['p_syms']) if cfg.get('p_syms') else []
    sym_table = {str(s): s for s in (list(x_syms) + list(p_syms))}
    dynamics = [sp.sympify(s, locals=sym_table) for s in cfg['dynamics']]
    barrier = sp.sympify(res['barrier'], locals=sym_table)

    p_values = cfg.get('p_values', {})
    P_lo = []
    P_hi = []
    if p_syms:
        for s in p_syms:
            v = p_values.get(str(s))
            # use parameter midpoint as both lo & hi if no explicit range
            P_lo.append(v if v is not None else 0.0)
            P_hi.append(v if v is not None else 0.0)

    out = verify_barrier(
        barrier=barrier, x_syms=list(x_syms), dynamics=dynamics,
        L_initial=cfg['L_initial'], U_initial=cfg['U_initial'],
        L_unsafe=cfg['L_unsafe'],   U_unsafe=cfg['U_unsafe'],
        L_space=cfg['L_space'],     U_space=cfg['U_space'],
        gamma=res['gamma'], lambda_=res['lambda'],
        p_syms=list(p_syms), P_lo=P_lo, P_hi=P_hi,
        tolerance=tolerance, timeout_s=timeout_s,
    )
    out['label'] = label
    return out


def main():
    results_dir = os.path.join(HERE, 'results')
    labels = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*.result.json'))):
        labels.append(os.path.basename(path).rsplit('.result.json', 1)[0])

    summary_lines = ['label                 | initial | lie     | unsafe-min            | overall']
    summary_lines.append('-' * 90)

    tolerance = float(os.environ.get('PROTECT_VERIFY_TOL', '0.0'))
    timeout_s = float(os.environ.get('PROTECT_VERIFY_TIMEOUT', '60'))

    # Optional label filter: PROTECT_VERIFY_LABELS=LALO20_W001,LALO20_W005,...
    label_filter = os.environ.get('PROTECT_VERIFY_LABELS', '').strip()
    if label_filter:
        wanted = set(label_filter.split(','))
        labels = [l for l in labels if l in wanted]
        print(f'  (filtered to {len(labels)} labels: {sorted(labels)})')

    for label in labels:
        res, cfg = load_pair(results_dir, label)
        if res is None or cfg is None:
            print(f'  {label}: no config side-file (skipped)')
            continue
        if 'barrier' not in res:
            print(f'  {label}: no barrier (NOT FOUND or TIMEOUT)')
            with open(os.path.join(results_dir, f'{label}.verify.json'), 'w') as f:
                json.dump({'label': label, 'overall': 'no-barrier'}, f, indent=2)
            summary_lines.append(f'{label:21s} | {"-":7s} | {"-":7s} | {"-":21s} | no-barrier')
            continue
        try:
            verdict = run_one(label, res, cfg,
                              tolerance=tolerance, timeout_s=timeout_s)
        except Exception as exc:
            print(f'  {label}: VERIFIER ERROR: {exc}')
            verdict = {'label': label, 'overall': 'error', 'error': str(exc)}
        with open(os.path.join(results_dir, f'{label}.verify.json'), 'w') as f:
            json.dump(verdict, f, indent=2, default=str)
        init  = verdict.get('initial', ('-',))[0]
        i_v   = verdict.get('initial', (None, None, None))[2]
        lie   = verdict.get('lie',     ('-',))[0]
        l_v   = verdict.get('lie',     (None, None, None))[2]
        u     = verdict.get('unsafe', [])
        u_str = ', '.join(t[0] for t in u) if u else '-'
        u_v   = ', '.join(f'{(t[2] if t[2] is not None else 0):.2g}' for t in u) if u else '-'
        overall = verdict.get('overall', '?')
        i_v_str = f'{i_v:.2g}' if i_v is not None else '-'
        l_v_str = f'{l_v:.2g}' if l_v is not None else '-'
        print(f'  {label}: init={init}({i_v_str}), lie={lie}({l_v_str}), unsafe={u_str} ({u_v}), overall={overall}')
        summary_lines.append(
            f'{label:21s} | init={init:7s} viol={i_v_str:9s} | lie={lie:7s} viol={l_v_str:9s} | unsafe={u_str:25s} viols={u_v:25s} | {overall}')

    summary_path = os.path.join(results_dir, 'verify_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f'\nSummary written to {summary_path}')


if __name__ == '__main__':
    main()
