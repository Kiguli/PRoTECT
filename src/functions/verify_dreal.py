"""
dReal-backed SMT verification (PRoTECT v2).

Same contract as verify_smt.verify_barrier, but emits SMT2 to a file
and runs dReal inside the dreal/dreal4 Docker image. dReal is a
delta-complete decision procedure for nonlinear real arithmetic --
much faster than Z3's nlsat on high-degree / high-dim queries (it
uses interval constraint propagation instead of polynomial nlsat).

The key trade-off is delta-completeness: dReal returns either:
  * delta-sat: a delta-cex point exists; the satisfying region has
    radius <= delta in each variable. (For verification, treat as a
    REAL counterexample when the inequality is also strict by more
    than delta.)
  * unsat: the formula is infeasible (sound).
  * timeout: indeterminate.

We use delta = 1e-4 (default), small enough to detect any meaningful
violation but large enough to avoid dReal getting stuck on the
machine-precision noise floor.
"""

import os
import shutil
import subprocess
import sympy as sp


DREAL_IMAGE = 'dreal/dreal4'


def _sympy_to_smt2(expr, sym_map):
    """Recursively convert a sympy polynomial expression to SMT-LIB
    prefix form. sym_map maps sympy.Symbol -> SMT-LIB var name."""
    expr = sp.sympify(expr)
    if expr.is_Number:
        if expr.is_Integer:
            v = int(expr)
            if v >= 0:
                return str(v)
            return f'(- {-v})'
        if expr.is_Rational:
            return f'(/ {int(expr.p)} {int(expr.q)})'
        # Floats: cast to rational.
        try:
            r = sp.Rational(float(expr)).limit_denominator(10 ** 9)
        except Exception:
            r = sp.Rational(str(expr))
        if int(r.q) == 1:
            v = int(r.p)
            return str(v) if v >= 0 else f'(- {-v})'
        if int(r.p) >= 0:
            return f'(/ {int(r.p)} {int(r.q)})'
        return f'(- (/ {-int(r.p)} {int(r.q)}))'
    if expr.is_Symbol:
        return sym_map[expr]
    if expr.is_Add:
        return '(+ ' + ' '.join(_sympy_to_smt2(a, sym_map) for a in expr.args) + ')'
    if expr.is_Mul:
        return '(* ' + ' '.join(_sympy_to_smt2(a, sym_map) for a in expr.args) + ')'
    if expr.is_Pow:
        base = expr.args[0]; exp = expr.args[1]
        if exp.is_Integer and int(exp) >= 0:
            n = int(exp)
            if n == 0:
                return '1'
            base_s = _sympy_to_smt2(base, sym_map)
            return '(* ' + ' '.join([base_s] * n) + ')'
        raise ValueError(f'non-integer exponent: {exp}')
    raise ValueError(f'unsupported sympy node: {type(expr).__name__}: {expr}')


def _emit_smt2(filename, decls, constraints):
    """Write an SMT2 file with QF_NRA logic, real-var declarations,
    and the given list of assertions."""
    with open(filename, 'w', encoding='ascii') as f:
        f.write('(set-logic QF_NRA)\n')
        for d in decls:
            f.write(d + '\n')
        for c in constraints:
            f.write(f'(assert {c})\n')
        f.write('(check-sat)\n')
        f.write('(exit)\n')


def _run_dreal(smt2_path, delta=1e-4, timeout_s=120, docker_image=DREAL_IMAGE):
    """Run dReal in Docker on smt2_path. Returns (status, output) where
    status is 'sat'|'unsat'|'unknown'|'timeout'|'error'."""
    smt2_path = os.path.abspath(smt2_path).replace('\\', '/')
    work_dir = os.path.dirname(smt2_path).replace('\\', '/')
    fname = os.path.basename(smt2_path)
    # Force MSYS path conversion off so the colon in -v survives.
    env = os.environ.copy()
    env['MSYS_NO_PATHCONV'] = '1'
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{work_dir}:/work',
        docker_image,
        'dreal',
        '--precision', str(delta),
        f'/work/{fname}',
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return 'timeout', ''
    out = (proc.stdout or '') + (proc.stderr or '')
    if 'delta-sat' in out:
        return 'sat', out
    if 'unsat' in out:
        return 'unsat', out
    return 'unknown', out


def _box(syms, lo, hi):
    """Return list of SMT2 inequality strings encoding lo[i] <= sym_i <= hi[i]."""
    out = []
    for s, l, h in zip(syms, lo, hi):
        l_r = sp.Rational(float(l)).limit_denominator(10 ** 9)
        h_r = sp.Rational(float(h)).limit_denominator(10 ** 9)
        l_str = (f'(/ {int(l_r.p)} {int(l_r.q)})'
                 if int(l_r.q) != 1 else str(int(l_r.p)))
        if int(l_r.p) < 0 and int(l_r.q) == 1:
            l_str = f'(- {-int(l_r.p)})'
        h_str = (f'(/ {int(h_r.p)} {int(h_r.q)})'
                 if int(h_r.q) != 1 else str(int(h_r.p)))
        if int(h_r.p) < 0 and int(h_r.q) == 1:
            h_str = f'(- {-int(h_r.p)})'
        out.append(f'(<= {l_str} {s})')
        out.append(f'(<= {s} {h_str})')
    return out


def verify_barrier_dreal(
    barrier, x_syms, dynamics,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    gamma, lambda_,
    p_syms=(), P_lo=None, P_hi=None,
    delta=1e-4,
    tolerance=0.0,
    timeout_s=180,
    work_dir=None,
):
    """dReal counterpart of verify_smt.verify_barrier. Returns a dict
    with the same shape (initial / unsafe / lie / overall).

    Status values are 'sat' / 'unsat' / 'timeout' / 'unknown'; sat means
    a delta-counterexample was found (treat as a true counterexample
    when the violation is well above delta)."""
    if shutil.which('docker') is None:
        return {'overall': 'error', 'error': 'docker not available'}
    work_dir = work_dir or os.path.join(os.path.dirname(__file__), '_dreal_tmp')
    os.makedirs(work_dir, exist_ok=True)

    sym_map = {}
    for s in x_syms:
        sym_map[s] = str(s)
    for s in p_syms:
        sym_map[s] = str(s)

    decls = []
    x_names = [sym_map[s] for s in x_syms]
    p_names = [sym_map[s] for s in p_syms]
    for n in x_names + p_names:
        decls.append(f'(declare-fun {n} () Real)')

    barrier_smt = _sympy_to_smt2(sp.expand(sp.sympify(barrier)), sym_map)
    gamma_r = sp.Rational(float(gamma)).limit_denominator(10 ** 9)
    lambda_r = sp.Rational(float(lambda_)).limit_denominator(10 ** 9)
    g_smt = (f'(/ {int(gamma_r.p)} {int(gamma_r.q)})'
             if int(gamma_r.q) != 1 else str(int(gamma_r.p)))
    if int(gamma_r.p) < 0 and int(gamma_r.q) == 1:
        g_smt = f'(- {-int(gamma_r.p)})'
    l_smt = (f'(/ {int(lambda_r.p)} {int(lambda_r.q)})'
             if int(lambda_r.q) != 1 else str(int(lambda_r.p)))
    if int(lambda_r.p) < 0 and int(lambda_r.q) == 1:
        l_smt = f'(- {-int(lambda_r.p)})'

    out = {'initial': None, 'unsafe': [], 'lie': None, 'delta': delta}

    # --- (i) initial ---
    file_i = os.path.join(work_dir, 'init.smt2')
    cs = _box(x_names, L_initial, U_initial) + [f'(> {barrier_smt} {g_smt})']
    _emit_smt2(file_i, decls, cs)
    out['initial'] = _run_dreal(file_i, delta=delta, timeout_s=timeout_s)

    # --- (ii) unsafe ---
    for j, (Lu, Uu) in enumerate(zip(L_unsafe, U_unsafe)):
        file_u = os.path.join(work_dir, f'unsafe_{j}.smt2')
        cs = _box(x_names, Lu, Uu) + [f'(< {barrier_smt} {l_smt})']
        _emit_smt2(file_u, decls, cs)
        out['unsafe'].append(_run_dreal(file_u, delta=delta, timeout_s=timeout_s))

    # --- (iii) lie ---
    grad = [sp.diff(sp.sympify(barrier), xi) for xi in x_syms]
    lie_expr = sum(grad[i] * sp.sympify(dynamics[i]) for i in range(len(x_syms)))
    lie_expr = sp.expand(lie_expr)
    lie_smt = _sympy_to_smt2(lie_expr, sym_map)
    file_l = os.path.join(work_dir, 'lie.smt2')
    cs = _box(x_names, L_space, U_space)
    if p_syms:
        cs += _box(p_names, P_lo, P_hi)
    cs.append(f'(> {lie_smt} 0)')
    _emit_smt2(file_l, decls, cs)
    out['lie'] = _run_dreal(file_l, delta=delta, timeout_s=timeout_s)

    statuses = ([out['initial'][0], out['lie'][0]] +
                [u[0] for u in out['unsafe']])
    if all(s == 'unsat' for s in statuses):
        out['overall'] = 'verified'
    elif any(s == 'sat' for s in statuses):
        out['overall'] = 'counterexample'
    else:
        out['overall'] = 'timeout-or-unknown'
    return out
