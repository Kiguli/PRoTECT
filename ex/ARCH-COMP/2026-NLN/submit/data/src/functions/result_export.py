"""
Per-benchmark side-channel writers.

  * write_result_json: barrier polynomial, gamma, lambda, b_degree
    (read by run_benchmarks.py to extend results.csv).
  * write_config_json: full problem setup (dynamics symbols, L/U arrays,
    projection list) so a stand-alone renderer can produce rich figures
    without re-importing the benchmark scripts.
"""

import json
import os

import numpy as np
import sympy as sp


def write_result_json(out_dir, label, result_dict):
    """
    Write a JSON file at ``out_dir/<label>.result.json`` capturing the
    barrier certificate produced by the SOS solver. Safe to call even
    when the solver returned an error (we just record the error).
    """
    os.makedirs(out_dir, exist_ok=True)
    payload = {}
    if 'barrier' in result_dict:
        payload['barrier'] = str(sp.sympify(result_dict['barrier']))
    if 'gamma' in result_dict:
        payload['gamma'] = float(result_dict['gamma'])
    if 'lambda' in result_dict:
        payload['lambda'] = float(result_dict['lambda'])
    if 'b_degree' in result_dict:
        payload['b_degree'] = int(result_dict['b_degree'])
    if 'error' in result_dict:
        payload['error'] = str(result_dict['error'])
    if 'solver' in result_dict:
        payload['solver'] = str(result_dict['solver'])
    if 'sos_overall' in result_dict:
        payload['sos_overall'] = str(result_dict['sos_overall'])
    if 'sos_residuals' in result_dict:
        payload['sos_residuals'] = {
            k: float(v) if isinstance(v, (int, float)) else str(v)
            for k, v in result_dict['sos_residuals'].items()
        }
    # Finite-time-specific & generic extra fields (optional).
    for k in ('time_orders', 'T_horizon', 'solve_time', 'solve_time_total',
              'variant'):
        if k in result_dict:
            v = result_dict[k]
            try:
                payload[k] = float(v) if isinstance(v, (int, float)) else v
            except Exception:
                payload[k] = str(v)
    if 'pointwise' in result_dict:
        pw = result_dict['pointwise']
        if isinstance(pw, dict):
            payload['pointwise'] = {
                k: (float(v) if isinstance(v, (int, float)) else str(v))
                for k, v in pw.items()
            }
        else:
            payload['pointwise'] = str(pw)
    out_path = os.path.join(out_dir, f'{label}.result.json')
    with open(out_path, 'w') as fp:
        json.dump(payload, fp, indent=2)
    return out_path


def _to_list(arr):
    return [float(v) for v in np.asarray(arr).flatten().tolist()]


def _array2_to_lists(arr):
    """List-of-lists for arrays of shape (n_regions, n_states)."""
    a = np.asarray(arr)
    if a.ndim == 1:
        return [_to_list(a)]
    return [[float(v) for v in row] for row in a]


def write_config_json(
    out_dir, label,
    x_syms, dynamics,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    projections,
    p_syms=(), p_values=None,
    title=''
):
    """Side-channel config file consumed by render_all.py.

    projections is a list of [dim_x, dim_y, x_label_tex, y_label_tex]
    tuples; the renderer makes one panel per projection.

    p_values is a dict mapping each parameter sympy symbol to a numeric
    value used for visualization (typically the midpoint of its box).
    """
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        'label': label,
        'title': title,
        'n_states': len(x_syms),
        'x_syms': [str(s) for s in x_syms],
        'dynamics': [str(sp.sympify(fi)) for fi in dynamics],
        'p_syms': [str(s) for s in p_syms],
        'p_values': {str(k): float(v) for k, v in (p_values or {}).items()},
        'L_initial': _to_list(L_initial),
        'U_initial': _to_list(U_initial),
        'L_unsafe':  _array2_to_lists(L_unsafe),
        'U_unsafe':  _array2_to_lists(U_unsafe),
        'L_space':   _to_list(L_space),
        'U_space':   _to_list(U_space),
        'projections': [list(t) for t in projections],
    }
    out_path = os.path.join(out_dir, f'{label}.config.json')
    with open(out_path, 'w') as fp:
        json.dump(payload, fp, indent=2)
    return out_path
