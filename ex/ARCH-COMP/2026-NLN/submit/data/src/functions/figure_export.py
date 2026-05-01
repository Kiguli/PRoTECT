"""
Tikz/latex-friendly figure exporter for the ARCH-COMP repeatability portal.

Beyond the original initial / unsafe set rectangles, this module now
also samples the barrier polynomial on a grid in the (dim_x, dim_y)
plane and emits contour polygons for the zero level set, the gamma
sub-level set, and the lambda super-level set. The resulting JSON has
all the visual elements needed to demonstrate the safety guarantee:

  * Initial set:    {x : ... } (filled white box)
  * Unsafe set(s):  {x : ... } (filled red boxes)
  * B(x) = 0    -- the canonical zero level set
  * B(x) = gamma -- the upper boundary of the safe sub-level set that
                   contains the initial set
  * B(x) = lambda-- the level at which the unsafe set is provably
                   excluded (lambda > gamma for a valid certificate)

Schema reference: see memory/arch_comp_nln_figure_format.md.
"""

import json
import os

import numpy as np
import sympy as sp


def _box_polygon(L, U, dim_x, dim_y):
    return {
        'x': [float(L[dim_x]), float(U[dim_x]), float(U[dim_x]),
              float(L[dim_x]), float(L[dim_x])],
        'y': [float(L[dim_y]), float(L[dim_y]), float(U[dim_y]),
              float(U[dim_y]), float(L[dim_y])],
    }


def _contour_polygons(barrier, x_syms, dim_x, dim_y,
                       L_space, U_space, levels, grid_n=120):
    """
    Sample the barrier polynomial on a grid in the (dim_x, dim_y) plane
    (other dims fixed at the midpoint of their state-space range) and
    return a dict ``{level: [{'x':..., 'y':...}, ...]}`` of contour
    polygons at each requested level.
    """
    others = {
        x_syms[i]: float((float(L_space[i]) + float(U_space[i])) / 2.0)
        for i in range(len(x_syms))
        if i != dim_x and i != dim_y
    }
    expr = sp.sympify(barrier).subs(others)
    f = sp.lambdify((x_syms[dim_x], x_syms[dim_y]), expr, 'numpy')

    xs = np.linspace(float(L_space[dim_x]), float(U_space[dim_x]), grid_n)
    ys = np.linspace(float(L_space[dim_y]), float(U_space[dim_y]), grid_n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.asarray(f(X, Y), dtype=float)

    # Use matplotlib only as a contour-extraction backend (no rendering).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = {}
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(X, Y, Z, levels=list(levels))
        for li, level in enumerate(cs.levels):
            polys = []
            try:
                # matplotlib < 3.8
                segs = cs.allsegs[li]
            except (AttributeError, IndexError):
                segs = []
            for seg in segs:
                arr = np.asarray(seg)
                if arr.size == 0 or arr.shape[0] < 2:
                    continue
                polys.append({
                    'x': [float(v) for v in arr[:, 0]],
                    'y': [float(v) for v in arr[:, 1]],
                })
            out[float(level)] = polys
    finally:
        plt.close(fig)
    return out


def export_figure(
    out_path,
    L_initial, U_initial,
    L_unsafe, U_unsafe,
    L_space, U_space,
    dim_x=0, dim_y=1,
    title='', x_label='x_{1}', y_label='x_{2}',
    trajectories=None,
    barrier=None, x_syms=None,
    gamma=None, lambda_=None,
):
    plot_data = []

    plot_data.append({
        'DisplayName': 'Initial set',
        'EdgeColor': [0, 0, 0],
        'FaceColor': [1, 1, 1],
        'Fill': True,
        'Data': [_box_polygon(L_initial, U_initial, dim_x, dim_y)],
    })

    plot_data.append({
        'DisplayName': 'Unsafe set',
        'EdgeColor': [0.9, 0.1, 0.1],
        'FaceColor': [0.9, 0.4, 0.4],
        'Fill': True,
        'Data': [_box_polygon(Lu, Uu, dim_x, dim_y)
                 for Lu, Uu in zip(L_unsafe, U_unsafe)],
    })

    # Optional barrier level sets. Build the level list dynamically so
    # we always include zero, plus gamma/lambda when known.
    if barrier is not None and x_syms is not None:
        levels = [0.0]
        if gamma is not None:
            levels.append(float(gamma))
        if lambda_ is not None:
            levels.append(float(lambda_))
        # de-dup / sort
        levels = sorted(set(round(L, 8) for L in levels))
        try:
            contours = _contour_polygons(
                barrier, x_syms, dim_x, dim_y,
                L_space, U_space, levels)
        except Exception as exc:
            contours = {}
            plot_data.append({
                'DisplayName': f'(contour extraction failed: {exc})',
                'Color': [0.5, 0.5, 0.5],
                'Fill': False,
                'Data': [],
            })

        # Distinct colour per level for clarity.
        palette = [
            ([0.10, 0.40, 0.85], 'B(x) = 0'),
            ([0.20, 0.65, 0.30], 'B(x) = gamma'),
            ([0.85, 0.55, 0.10], 'B(x) = lambda'),
        ]
        for idx, level in enumerate(levels):
            polys = contours.get(level, [])
            if not polys:
                continue
            colour, _ = palette[min(idx, len(palette) - 1)]
            if level == 0.0:
                name = 'B(x) = 0'
            elif gamma is not None and abs(level - float(gamma)) < 1e-7:
                name = f'B(x) = gamma ({level:.3g})'
            elif lambda_ is not None and abs(level - float(lambda_)) < 1e-7:
                name = f'B(x) = lambda ({level:.3g})'
            else:
                name = f'B(x) = {level:.3g}'
            plot_data.append({
                'DisplayName': name,
                'Color': colour,
                'EdgeColor': colour,
                'Fill': False,
                'Data': polys,
            })

    if trajectories:
        plot_data.append({
            'DisplayName': 'Simulations',
            'Color': [0, 0, 0],
            'Fill': False,
            'Data': [{'x': list(xs), 'y': list(ys)}
                     for xs, ys in trajectories],
        })

    figure = {
        'Title': title,
        'XLim': [float(L_space[dim_x]), float(U_space[dim_x])],
        'YLim': [float(L_space[dim_y]), float(U_space[dim_y])],
        'XLabel': x_label,
        'YLabel': y_label,
        'PlotData': plot_data,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fp:
        json.dump(figure, fp, indent=2)
