"""
Multi-panel matplotlib renderer.

For each benchmark in submit/results/<label>.config.json (paired with
<label>.result.json), produce a single PNG with one sub-panel per
projection. Each panel shows:

  * Heatmap of B(x, y, others_fixed)   -- shape of the certificate.
  * Many thin contour lines of B       -- detail (gamma=lambda usually
                                          collapse to a single black
                                          contour, but the surrounding
                                          contours show the level
                                          structure).
  * Vector field of f(x)               -- normalized arrows.
  * Simulated trajectories from corners of the initial set.
  * Initial set                        -- cyan box.
  * Unsafe set(s) projected to this plane (filtered: skip boxes whose
    constraint is purely off-axis).

Run:
    python render_all.py
"""

import glob
import json
import os
import sys

import numpy as np
import sympy as sp


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.integrate import odeint


def _filter_unsafe(L_unsafe, U_unsafe, L_space, U_space, dim_x, dim_y):
    keep = []
    for Lu, Uu in zip(L_unsafe, U_unsafe):
        sx = (Lu[dim_x] > L_space[dim_x] + 1e-9 or
              Uu[dim_x] < U_space[dim_x] - 1e-9)
        sy = (Lu[dim_y] > L_space[dim_y] + 1e-9 or
              Uu[dim_y] < U_space[dim_y] - 1e-9)
        if sx or sy:
            keep.append((Lu, Uu))
    return keep


def _midpoint_subs(x_syms, dim_x, dim_y, L_space, U_space, p_subs):
    subs = dict(p_subs or {})
    for i, sym in enumerate(x_syms):
        if i == dim_x or i == dim_y:
            continue
        subs[sym] = float((float(L_space[i]) + float(U_space[i])) / 2.0)
    return subs


def _simulate(x_syms, dynamics, p_subs, L_initial, U_initial, dim_x, dim_y,
              T=10.0, n_steps=400, max_traj=4):
    n = len(x_syms)
    f_subbed = [sp.sympify(fi).subs(p_subs or {}) for fi in dynamics]
    f_lambdified = sp.lambdify(list(x_syms), f_subbed, 'numpy')

    def rhs(state, t):
        return list(f_lambdified(*state))

    cx_lo = float(L_initial[dim_x]); cx_hi = float(U_initial[dim_x])
    cy_lo = float(L_initial[dim_y]); cy_hi = float(U_initial[dim_y])
    seeds = [(cx_lo, cy_lo), (cx_hi, cy_hi), (cx_lo, cy_hi), (cx_hi, cy_lo)][:max_traj]
    t = np.linspace(0.0, float(T), n_steps)
    out = []
    for cx, cy in seeds:
        x0 = []
        for i in range(n):
            if i == dim_x: x0.append(cx)
            elif i == dim_y: x0.append(cy)
            else: x0.append(float((float(L_initial[i]) + float(U_initial[i])) / 2.0))
        try:
            sol = odeint(rhs, x0, t, full_output=False, mxstep=2000,
                         rtol=1e-6, atol=1e-9)
            out.append((sol[:, dim_x], sol[:, dim_y]))
        except Exception:
            continue
    return out


def render_panel(ax, cfg, res, projection, x_syms, p_subs):
    dim_x, dim_y = int(projection[0]), int(projection[1])
    x_label = projection[2] if len(projection) > 2 else f'x_{dim_x}'
    y_label = projection[3] if len(projection) > 3 else f'x_{dim_y}'

    L_space = cfg['L_space']; U_space = cfg['U_space']
    L_initial = cfg['L_initial']; U_initial = cfg['U_initial']
    L_unsafe = cfg['L_unsafe']; U_unsafe = cfg['U_unsafe']
    dynamics = [sp.sympify(s) for s in cfg['dynamics']]

    ax.set_xlim(float(L_space[dim_x]), float(U_space[dim_x]))
    ax.set_ylim(float(L_space[dim_y]), float(U_space[dim_y]))

    # ---- barrier heatmap + many contour lines ----
    if res and 'barrier' in res:
        try:
            barrier = sp.sympify(res['barrier'])
            subs = _midpoint_subs(x_syms, dim_x, dim_y, L_space, U_space, p_subs)
            b_proj = barrier.subs(subs)
            f_b = sp.lambdify((x_syms[dim_x], x_syms[dim_y]), b_proj, 'numpy')
            xs = np.linspace(float(L_space[dim_x]), float(U_space[dim_x]), 140)
            ys = np.linspace(float(L_space[dim_y]), float(U_space[dim_y]), 140)
            X, Y = np.meshgrid(xs, ys)
            Z = np.asarray(f_b(X, Y), dtype=float)
            zmin = float(np.nanmin(Z)); zmax = float(np.nanmax(Z))
            if zmax > zmin:
                cf = ax.contourf(X, Y, Z, levels=22, cmap='viridis', alpha=0.55)
                level_grid = np.linspace(zmin, zmax, 14)
                ax.contour(X, Y, Z, levels=level_grid, colors='white',
                           linewidths=0.4, alpha=0.55)
                gamma = res.get('gamma'); lam = res.get('lambda')
                if gamma is not None and zmin <= gamma <= zmax:
                    ax.contour(X, Y, Z, levels=[float(gamma)],
                               colors='black', linewidths=2.0, linestyles='-')
                if lam is not None and zmin <= lam <= zmax \
                        and (gamma is None or abs(lam - gamma) > 1e-5):
                    ax.contour(X, Y, Z, levels=[float(lam)],
                               colors='black', linewidths=2.0, linestyles='--')
        except Exception:
            pass

    # ---- vector field (sparse) ----
    try:
        f_subbed_full = [sp.sympify(fi).subs(_midpoint_subs(
            x_syms, dim_x, dim_y, L_space, U_space, p_subs))
                         for fi in dynamics]
        fx_l = sp.lambdify((x_syms[dim_x], x_syms[dim_y]),
                           f_subbed_full[dim_x], 'numpy')
        fy_l = sp.lambdify((x_syms[dim_x], x_syms[dim_y]),
                           f_subbed_full[dim_y], 'numpy')
        xs = np.linspace(float(L_space[dim_x]), float(U_space[dim_x]), 18)
        ys = np.linspace(float(L_space[dim_y]), float(U_space[dim_y]), 18)
        Xq, Yq = np.meshgrid(xs, ys)
        U = np.asarray(fx_l(Xq, Yq), dtype=float) * np.ones_like(Xq)
        V = np.asarray(fy_l(Xq, Yq), dtype=float) * np.ones_like(Yq)
        mag = np.sqrt(U * U + V * V); mag[mag < 1e-12] = 1.0
        ax.quiver(Xq, Yq, U / mag, V / mag,
                  angles='xy', scale_units='xy', scale=12,
                  color='#222', alpha=0.65, width=0.0028,
                  headwidth=4.0, headlength=5.0)
    except Exception:
        pass

    # ---- trajectories from initial corners ----
    try:
        trajs = _simulate(x_syms, dynamics, p_subs,
                          L_initial, U_initial, dim_x, dim_y,
                          T=10.0, n_steps=300, max_traj=4)
        for xs_t, ys_t in trajs:
            ax.plot(xs_t, ys_t, color='white', linewidth=2.4,
                    alpha=0.95, zorder=4)
            ax.plot(xs_t, ys_t, color='#0a4', linewidth=1.2,
                    alpha=0.95, zorder=5)
    except Exception:
        pass

    # ---- initial set ----
    rect = Rectangle(
        (float(L_initial[dim_x]), float(L_initial[dim_y])),
        float(U_initial[dim_x] - L_initial[dim_x]),
        float(U_initial[dim_y] - L_initial[dim_y]),
        linewidth=2.0, edgecolor='cyan', facecolor='cyan', alpha=0.6,
        zorder=6)
    ax.add_patch(rect)

    # ---- unsafe sets, filtered to those non-trivial in this projection ----
    kept = _filter_unsafe(L_unsafe, U_unsafe, L_space, U_space, dim_x, dim_y)
    skipped = len(L_unsafe) - len(kept)
    for Lu, Uu in kept:
        rect = Rectangle(
            (float(Lu[dim_x]), float(Lu[dim_y])),
            float(Uu[dim_x] - Lu[dim_x]),
            float(Uu[dim_y] - Lu[dim_y]),
            linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.45,
            hatch='///', zorder=6)
        ax.add_patch(rect)
    if skipped:
        ax.text(0.99, 0.02, f'(+{skipped} off-axis unsafe)',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, color='#933', style='italic', alpha=0.7)

    ax.set_xlabel(f'${x_label}$' if x_label else '')
    ax.set_ylabel(f'${y_label}$' if y_label else '')
    ax.tick_params(labelsize=8)
    ax.grid(False)


def render_one(label, results_dir):
    cfg_path = os.path.join(results_dir, f'{label}.config.json')
    res_path = os.path.join(results_dir, f'{label}.result.json')
    if not os.path.isfile(cfg_path):
        print(f'  {label}: skipped (no config)')
        return False
    with open(cfg_path) as f: cfg = json.load(f)
    res = {}
    if os.path.isfile(res_path):
        with open(res_path) as f: res = json.load(f)

    x_syms = sp.symbols(cfg['x_syms'])
    if isinstance(x_syms, sp.Symbol):
        x_syms = (x_syms,)
    x_syms = tuple(x_syms)
    p_syms_names = cfg.get('p_syms', [])
    p_syms = sp.symbols(p_syms_names) if p_syms_names else ()
    if isinstance(p_syms, sp.Symbol):
        p_syms = (p_syms,)
    p_subs = {}
    pv = cfg.get('p_values', {})
    for s in p_syms:
        if str(s) in pv:
            p_subs[s] = float(pv[str(s)])

    projections = cfg.get('projections', [[0, 1, 'x_1', 'x_2']])
    n_panels = len(projections)
    cols = min(3, n_panels)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.0 * rows))
    if n_panels == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[a] for a in axes])

    for k, proj in enumerate(projections):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        try:
            render_panel(ax, cfg, res, proj, x_syms, p_subs)
        except Exception as exc:
            ax.text(0.5, 0.5, f'render failed: {exc}',
                    transform=ax.transAxes, ha='center', va='center',
                    color='red', fontsize=9)

    # Hide unused axes.
    for k in range(n_panels, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis('off')

    title = cfg.get('title', label)
    if 'barrier' in res:
        gamma = res.get('gamma'); lam = res.get('lambda')
        bd = res.get('b_degree')
        title += f'   |   degree={bd}, γ={gamma:.3g}, λ={lam:.3g}'
    else:
        title += '   |   barrier NOT FOUND'
    fig.suptitle(title, fontsize=12, y=1.005)

    out_path = os.path.join(results_dir, f'{label}.png')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  {label}: {n_panels} panels -> {out_path}')
    return True


def main():
    results_dir = os.path.join(HERE, 'results')
    labels = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*.config.json'))):
        labels.append(os.path.basename(path).rsplit('.config.json', 1)[0])

    label_filter = os.environ.get('PROTECT_RENDER_LABELS', '').strip()
    if label_filter:
        wanted = set(label_filter.split(','))
        labels = [l for l in labels if l in wanted]

    print(f'rendering {len(labels)} benchmarks ...')
    for label in labels:
        try:
            render_one(label, results_dir)
        except Exception as exc:
            print(f'  {label}: FAILED ({exc})')


if __name__ == '__main__':
    main()
