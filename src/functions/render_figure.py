"""
Rich matplotlib PNG renderer for PRoTECT v2 barrier-certificate
visualisations. Designed to make the safety guarantee intuitive:

  * Heatmap of B(x) over the (dim_x, dim_y) projection plane so the
    certificate's shape is visible (other dims fixed at the midpoint
    of the state-space envelope; parametric variables fixed at the
    midpoint of their box).
  * Many contour lines spanning min/max(B) on the window, with B = gamma
    and B = lambda highlighted as thick black lines (the safety
    boundaries).
  * Vector field of f(x) as a quiver overlay (sparse grid).
  * Simulated trajectories from corner points of the initial set,
    integrated for `time_horizon` seconds via scipy.integrate.odeint.
  * Initial set (white box, thick border).
  * Unsafe set(s), filtered to skip "off-axis" boxes that fill the
    projection window because their constraint is on dimensions not
    plotted.

Note on PRoTECT's level-set convention: B is SOS-non-negative, so
B = 0 is rarely a continuous boundary. The meaningful safety lines are
gamma (initial-set containment) and lambda (unsafe-set exclusion); the
heatmap + many contours communicate the rest of the certificate's
geometry.
"""

import os

import numpy as np
import sympy as sp


def _project_substitutions(x_syms, dim_x, dim_y, L_space, U_space, p_subs):
    """Build a sympy substitution dict that fixes every state variable
    EXCEPT the two projection axes at the midpoint of its state-space
    range, and applies any parameter substitutions."""
    subs = dict(p_subs or {})
    for i, sym in enumerate(x_syms):
        if i == dim_x or i == dim_y:
            continue
        mid = float((float(L_space[i]) + float(U_space[i])) / 2.0)
        subs[sym] = mid
    return subs


def _filter_unsafe(L_unsafe, U_unsafe, L_space, U_space, dim_x, dim_y):
    """Keep only unsafe boxes whose projection onto (dim_x, dim_y) is a
    proper sub-rectangle of the state-space window. Drop boxes that
    cover the entire window in both projection axes (their constraint
    is purely off-axis and visually meaningless in this projection)."""
    keep = []
    for Lu, Uu in zip(L_unsafe, U_unsafe):
        sx = (Lu[dim_x] > L_space[dim_x] + 1e-9 or
              Uu[dim_x] < U_space[dim_x] - 1e-9)
        sy = (Lu[dim_y] > L_space[dim_y] + 1e-9 or
              Uu[dim_y] < U_space[dim_y] - 1e-9)
        if sx or sy:
            keep.append((Lu, Uu))
    return keep


def _simulate_trajectories(x_syms, dynamics, p_subs, L_initial, U_initial,
                           L_space, U_space, dim_x, dim_y, time_horizon,
                           n_corner=4, n_steps=400):
    """Integrate dynamics from a few initial-set corner points and
    return a list of (xs, ys) arrays, one per trajectory, projected
    onto (dim_x, dim_y)."""
    from scipy.integrate import odeint

    n = len(x_syms)
    f_subbed = [sp.sympify(fi).subs(p_subs or {}) for fi in dynamics]
    f_lambdified = sp.lambdify(list(x_syms), f_subbed, 'numpy')

    def rhs(state, t):
        return list(f_lambdified(*state))

    # Sample n_corner corner-ish points: take 2D corners on (dim_x, dim_y)
    # and centroid for other dims.
    corners_2d = [
        (float(L_initial[dim_x]), float(L_initial[dim_y])),
        (float(U_initial[dim_x]), float(L_initial[dim_y])),
        (float(L_initial[dim_x]), float(U_initial[dim_y])),
        (float(U_initial[dim_x]), float(U_initial[dim_y])),
    ][:n_corner]

    t = np.linspace(0.0, float(time_horizon), n_steps)
    out = []
    for cx, cy in corners_2d:
        x0 = []
        for i in range(n):
            if i == dim_x:
                x0.append(cx)
            elif i == dim_y:
                x0.append(cy)
            else:
                x0.append(float((float(L_initial[i]) + float(U_initial[i])) / 2.0))
        try:
            sol = odeint(rhs, x0, t, full_output=False, mxstep=5000)
            # Clip trajectory to the state-space window (visual, not safe).
            out.append((sol[:, dim_x], sol[:, dim_y]))
        except Exception:
            continue
    return out


def render_png(
    out_path,
    x_syms, dynamics,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    barrier=None, gamma=None, lambda_=None,
    p_subs=None,
    dim_x=0, dim_y=1,
    title='', x_label='x_1', y_label='x_2',
    time_horizon=10.0,
    grid_n=140, vector_n=22, n_levels=18,
):
    """
    Write a rich PNG visualisation. Safe to call on a benchmark that
    failed to find a barrier (in that case the heatmap and level sets
    are skipped but the rest is still shown).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    proj_subs = _project_substitutions(x_syms, dim_x, dim_y, L_space, U_space, p_subs)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(float(L_space[dim_x]), float(U_space[dim_x]))
    ax.set_ylim(float(L_space[dim_y]), float(U_space[dim_y]))

    # --- barrier heatmap + many contour lines ---
    if barrier is not None:
        try:
            b_proj = sp.sympify(barrier).subs(proj_subs)
            f_b = sp.lambdify(
                (x_syms[dim_x], x_syms[dim_y]), b_proj, 'numpy')
            xs = np.linspace(float(L_space[dim_x]), float(U_space[dim_x]), grid_n)
            ys = np.linspace(float(L_space[dim_y]), float(U_space[dim_y]), grid_n)
            X, Y = np.meshgrid(xs, ys)
            Z = np.asarray(f_b(X, Y), dtype=float)

            zmin = float(np.nanmin(Z))
            zmax = float(np.nanmax(Z))
            if zmax > zmin:
                cf = ax.contourf(X, Y, Z, levels=24, cmap='viridis', alpha=0.55)
                cb = plt.colorbar(cf, ax=ax, shrink=0.85,
                                  label='B(x) on projection (other dims fixed)')

                # Many thin level lines for shape; gamma/lambda thick.
                level_grid = np.linspace(zmin, zmax, n_levels)
                ax.contour(X, Y, Z, levels=level_grid,
                           colors='white', linewidths=0.4, alpha=0.7)

                if gamma is not None and zmin <= float(gamma) <= zmax:
                    cs_g = ax.contour(X, Y, Z, levels=[float(gamma)],
                                       colors='black', linewidths=2.4,
                                       linestyles='-')
                    ax.clabel(cs_g, fmt={float(gamma): f'B = γ = {float(gamma):.3g}'},
                              fontsize=8, inline=True)
                if lambda_ is not None and zmin <= float(lambda_) <= zmax \
                        and (gamma is None or abs(float(lambda_) - float(gamma)) > 1e-5):
                    cs_l = ax.contour(X, Y, Z, levels=[float(lambda_)],
                                       colors='black', linewidths=2.4,
                                       linestyles='--')
                    ax.clabel(cs_l, fmt={float(lambda_): f'B = λ = {float(lambda_):.3g}'},
                              fontsize=8, inline=True)
        except Exception as exc:
            ax.text(0.02, 0.98,
                    f'(barrier render failed: {exc})',
                    transform=ax.transAxes, va='top', fontsize=8,
                    color='red')

    # --- vector field (quiver) ---
    if dynamics is not None:
        try:
            f_subbed = [sp.sympify(fi).subs(proj_subs) for fi in dynamics]
            fx = sp.lambdify(
                (x_syms[dim_x], x_syms[dim_y]), f_subbed[dim_x], 'numpy')
            fy = sp.lambdify(
                (x_syms[dim_x], x_syms[dim_y]), f_subbed[dim_y], 'numpy')
            xs = np.linspace(float(L_space[dim_x]), float(U_space[dim_x]), vector_n)
            ys = np.linspace(float(L_space[dim_y]), float(U_space[dim_y]), vector_n)
            Xq, Yq = np.meshgrid(xs, ys)
            U = np.asarray(fx(Xq, Yq), dtype=float)
            V = np.asarray(fy(Xq, Yq), dtype=float)
            # Normalise per-arrow so direction is visible regardless of magnitude.
            mag = np.sqrt(U * U + V * V)
            mag[mag < 1e-12] = 1.0
            ax.quiver(Xq, Yq, U / mag, V / mag,
                      angles='xy', scale_units='xy', scale=vector_n / 1.6,
                      color='#222', alpha=0.7, width=0.0025,
                      headwidth=4, headlength=5)
        except Exception:
            pass

    # --- simulated trajectories ---
    if dynamics is not None:
        try:
            trajs = _simulate_trajectories(
                x_syms, dynamics, p_subs,
                L_initial, U_initial, L_space, U_space,
                dim_x, dim_y, time_horizon)
            for xs, ys in trajs:
                ax.plot(xs, ys, color='white', linewidth=2.0, alpha=0.95,
                        zorder=4)
                ax.plot(xs, ys, color='#0a4', linewidth=0.9, alpha=0.95,
                        zorder=5, label=None)
            if trajs:
                # one legend entry
                ax.plot([], [], color='#0a4', linewidth=2.0,
                        label=f'Trajectories ({len(trajs)})')
        except Exception:
            pass

    # --- initial set ---
    rect = Rectangle(
        (float(L_initial[dim_x]), float(L_initial[dim_y])),
        float(U_initial[dim_x] - L_initial[dim_x]),
        float(U_initial[dim_y] - L_initial[dim_y]),
        linewidth=2.5, edgecolor='cyan', facecolor='cyan', alpha=0.55,
        label='Initial set', zorder=6)
    ax.add_patch(rect)

    # --- unsafe sets (filtered) ---
    kept = _filter_unsafe(L_unsafe, U_unsafe, L_space, U_space, dim_x, dim_y)
    skipped = len(L_unsafe) - len(kept)
    for Lu, Uu in kept:
        rect = Rectangle(
            (float(Lu[dim_x]), float(Lu[dim_y])),
            float(Uu[dim_x] - Lu[dim_x]),
            float(Uu[dim_y] - Lu[dim_y]),
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.55,
            hatch='///', zorder=6)
        ax.add_patch(rect)
    if kept:
        ax.add_patch(Rectangle((0, 0), 0, 0, edgecolor='red',
                               facecolor='red', alpha=0.55, hatch='///',
                               label='Unsafe set'))
    if skipped:
        ax.text(0.99, 0.01,
                f'({skipped} unsafe region(s) hidden -- off-axis in this projection)',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='#933', style='italic')

    ax.set_xlabel(f'${x_label}$' if x_label else '')
    ax.set_ylabel(f'${y_label}$' if y_label else '')
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
