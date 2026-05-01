"""
Hard-code the per-benchmark config (dynamics + L/U arrays + parameter
midpoints) and dump as <label>.config.json files into submit/results/.
Side-effect: the next verify_all.py / render_all.py runs can consume
these without re-running the SOS solver.

This is a one-time bridge: the benchmark scripts will eventually call
result_export.write_config_json themselves, but rerunning the full
Docker sweep takes ~25 min, so we mirror the data here for now.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)


def _box_unsafe(i, threshold, sense, L_space, U_space, n):
    """Return (Lu, Uu) where dim i is bounded by `threshold` from below
    or above, and other dims span the state space."""
    Lu = list(L_space); Uu = list(U_space)
    if sense == '>=':
        Lu[i] = threshold
    elif sense == '<=':
        Uu[i] = threshold
    else:
        raise ValueError(sense)
    return Lu, Uu


def write(label, **fields):
    fields['label'] = label
    out = os.path.join(RESULTS, f'{label}.config.json')
    with open(out, 'w') as f:
        json.dump(fields, f, indent=2)
    print(f'  {out}')


# ----------------------------------------------------------------------
# LOVO25 -- Lotka-Volterra, 2-D
# ----------------------------------------------------------------------
write('LOVO25',
    title='LOVO25 (Lotka-Volterra)',
    n_states=2,
    x_syms=['x0', 'x1'],
    dynamics=['3*x0 - 3*x0*x1', 'x0*x1 - x1'],
    p_syms=[], p_values={},
    L_initial=[1.288, 0.999999], U_initial=[1.312, 1.000001],
    L_space=[0.5, 0.5], U_space=[1.5, 1.5],
    L_unsafe=[
        [0.5, 0.5], [1.4, 0.5], [0.5, 0.5], [0.5, 1.4],
    ],
    U_unsafe=[
        [0.6, 1.5], [1.5, 1.5], [1.5, 0.6], [1.5, 1.5],
    ],
    projections=[[0, 1, 'x', 'y']],
)


# ----------------------------------------------------------------------
# LOVO21 -- Lorenz, 3-D
# ----------------------------------------------------------------------
write('LOVO21',
    title='LOVO21 (Lorenz, original spec)',
    n_states=3,
    x_syms=['x0', 'x1', 'x2'],
    dynamics=[
        '10.0*x1 - 10.0*x0',
        'x0*(28.0 - x2) - x1',
        'x0*x1 - 8/3*x2',
    ],
    p_syms=[], p_values={},
    L_initial=[0.9, -0.01, -0.01], U_initial=[1.1, 0.01, 0.01],
    L_space=[-30.0, -30.0, 0.0], U_space=[30.0, 30.0, 60.0],
    L_unsafe=[[20.0, -30.0, 0.0]],
    U_unsafe=[[30.0, 30.0, 60.0]],
    projections=[[0, 1, 'x', 'y'], [0, 2, 'x', 'z'], [1, 2, 'y', 'z']],
)


# ----------------------------------------------------------------------
# ROBE21 / ROBE25 -- rescaled Robertson kinetics, 3-D
# Both share the same rescaled dynamics; only initial-set radius differs.
# ----------------------------------------------------------------------
SCALE_Y = 1e4
def robe21_config(eps, label):
    write(label,
        title=f'{label} (Robertson, rescaled)',
        n_states=3,
        x_syms=['x0', 'x1', 'x2'],
        # u' = -0.04*u + V*w
        # V' = 400*u - 1e4*V*w - 300*V^2
        # w' = 0.3*V^2
        dynamics=[
            '-0.04*x0 + x1*x2',
            '400*x0 - 10000*x1*x2 - 300*x1**2',
            '0.3*x1**2',
        ],
        p_syms=[], p_values={},
        L_initial=[1.0 - eps, 0.0, 0.0],
        U_initial=[1.0 + eps, SCALE_Y * eps, eps],
        L_space=[0.0, 0.0, 0.0],
        U_space=[1.1, SCALE_Y * 0.01, 1.1],
        L_unsafe=[
            [0.0, 0.0, 0.0],          # u <= 0.9
            [0.0, 0.0, 1.0],          # w >= 1.0
        ],
        U_unsafe=[
            [0.9, SCALE_Y * 0.01, 1.1],
            [1.1, SCALE_Y * 0.01, 1.1],
        ],
        projections=[[0, 2, 'u', 'w'], [0, 1, 'u', 'V']],
    )

robe21_config(0.001, 'ROBE21_1')
robe21_config(0.005, 'ROBE21_2')
robe21_config(0.01,  'ROBE21_3')

# ROBE25 same setup, instances 2/3 use larger eps and shifted scale_y.
# Only instance 1 was solved cleanly with the same rescaling as ROBE21;
# instances 2, 3 used different scale factors but our run used the
# same ROBE21 setup at 0.001 / 0.005 / 0.01 inside the running suite.
robe21_config(0.001, 'ROBE25_1')
robe21_config(0.005, 'ROBE25_2')
robe21_config(0.01,  'ROBE25_3')


# ----------------------------------------------------------------------
# CVDP22 -- Coupled van der Pol, b=70, 4-D
# ----------------------------------------------------------------------
write('CVDP22',
    title='CVDP22 (b=70 fixed)',
    n_states=4,
    x_syms=['x0', 'x1', 'x2', 'x3'],
    dynamics=[
        'x1',
        '(1 - x0**2)*x1 + 70*(x2 - x0) - x0',
        'x3',
        '(1 - x2**2)*x3 - 70*(x2 - x0) - x2',
    ],
    p_syms=[], p_values={},
    L_initial=[1.25, 2.35, 1.25, 2.35],
    U_initial=[1.55, 2.45, 1.55, 2.45],
    L_space=[-3.0, -4.0, -3.0, -4.0],
    U_space=[ 3.0,  4.0,  3.0,  4.0],
    L_unsafe=[
        [-3.0,  3.7, -3.0, -4.0],
        [-3.0, -4.0, -3.0,  3.7],
    ],
    U_unsafe=[
        [3.0, 4.0, 3.0, 4.0],
        [3.0, 4.0, 3.0, 4.0],
    ],
    projections=[
        [0, 1, 'x_1', 'y_1'],
        [2, 3, 'x_2', 'y_2'],
        [0, 2, 'x_1', 'x_2'],
        [1, 3, 'y_1', 'y_2'],
    ],
)


# ----------------------------------------------------------------------
# CVDP23 (b=2 fixed) -- same shape as CVDP22 but b=2 and unsafe y>=2.75
# ----------------------------------------------------------------------
write('CVDP23_b2',
    title='CVDP23 (b=2 fixed-midpoint reduction)',
    n_states=4,
    x_syms=['x0', 'x1', 'x2', 'x3'],
    dynamics=[
        'x1',
        '(1 - x0**2)*x1 + 2*(x2 - x0) - x0',
        'x3',
        '(1 - x2**2)*x3 - 2*(x2 - x0) - x2',
    ],
    p_syms=[], p_values={},
    L_initial=[1.25, 2.35, 1.25, 2.35],
    U_initial=[1.55, 2.45, 1.55, 2.45],
    L_space=[-3.0, -3.0, -3.0, -3.0],
    U_space=[ 3.0,  3.0,  3.0,  3.0],
    L_unsafe=[
        [-3.0,  2.75, -3.0, -3.0],
        [-3.0, -3.0,  -3.0,  2.75],
    ],
    U_unsafe=[
        [3.0, 3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0, 3.0],
    ],
    projections=[
        [0, 1, 'x_1', 'y_1'],
        [2, 3, 'x_2', 'y_2'],
    ],
)


# ----------------------------------------------------------------------
# CVDP23_b_unc -- b in [1, 3] uncertain (parameter), 4-D + 1 param
# ----------------------------------------------------------------------
write('CVDP23_b_unc',
    title='CVDP23 (b in [1,3] uncertain, v2 robust SOS)',
    n_states=4,
    x_syms=['x0', 'x1', 'x2', 'x3'],
    dynamics=[
        'x1',
        '(1 - x0**2)*x1 + b0*(x2 - x0) - x0',
        'x3',
        '(1 - x2**2)*x3 - b0*(x2 - x0) - x2',
    ],
    p_syms=['b0'], p_values={'b0': 2.0},
    L_initial=[1.25, 2.35, 1.25, 2.35],
    U_initial=[1.55, 2.45, 1.55, 2.45],
    L_space=[-3.0, -3.0, -3.0, -3.0],
    U_space=[ 3.0,  3.0,  3.0,  3.0],
    L_unsafe=[
        [-3.0,  2.75, -3.0, -3.0],
        [-3.0, -3.0,  -3.0,  2.75],
    ],
    U_unsafe=[
        [3.0, 3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0, 3.0],
    ],
    projections=[
        [0, 1, 'x_1', 'y_1'],
        [2, 3, 'x_2', 'y_2'],
    ],
)


# ----------------------------------------------------------------------
# LALO20 -- Laub-Loomis 7-D enzymatic
# ----------------------------------------------------------------------
def lalo_config(W, unsafe_x4, label):
    center = [1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45]
    L_init = [c - W for c in center]; U_init = [c + W for c in center]
    L_space = [0.5, 0.5, 1.0, 1.5, 0.5, 0.05, 0.2]
    U_space = [2.5, 2.5, 4.0, 6.0, 2.5, 0.5, 1.2]
    Lu = list(L_space); Uu = list(U_space); Lu[3] = unsafe_x4
    write(label,
        title=f'LALO20 ({label[7:]})',
        n_states=7,
        x_syms=[f'x{i}' for i in range(7)],
        dynamics=[
            '1.4*x2 - 0.9*x0',
            '2.5*x4 - 1.5*x1',
            '0.6*x6 - 0.8*x1*x2',
            '2 - 1.3*x2*x3',
            '0.7*x0 - x3*x4',
            '0.3*x0 - 3.1*x5',
            '1.8*x5 - 1.5*x1*x6',
        ],
        p_syms=[], p_values={},
        L_initial=L_init, U_initial=U_init,
        L_space=L_space, U_space=U_space,
        L_unsafe=[Lu], U_unsafe=[Uu],
        projections=[
            [0, 3, 'x_1', 'x_4'],
            [1, 3, 'x_2', 'x_4'],
            [2, 3, 'x_3', 'x_4'],
            [4, 3, 'x_5', 'x_4'],
            [5, 3, 'x_6', 'x_4'],
            [6, 3, 'x_7', 'x_4'],
        ],
    )

lalo_config(0.01, 4.5, 'LALO20_W001')
lalo_config(0.05, 4.5, 'LALO20_W005')
lalo_config(0.10, 5.0, 'LALO20_W01')


# ----------------------------------------------------------------------
# SPRE22 -- Spacecraft rendezvous, polynomialised gravity, 4-D
# ----------------------------------------------------------------------
r = 42164e3
mu_ = 3.986e14 * 60**2
m_c = 500.0
n   = (mu_ / r**3)**0.5
g_lin = mu_ / r**3
K1 = [
    [-28.8287,   0.1005, -1449.9754,    0.0046],
    [ -0.087,  -33.2562,     0.00462, -1451.5013],
]

def _spre22_dyn():
    """Return the 4 polynomial dynamics expressions as strings for SPRE22."""
    f1 = 'x2'
    f2 = 'x3'
    f3 = (
        f'({n**2 + K1[0][0]/m_c})*x0 + '
        f'({2*n + K1[0][3]/m_c})*x3 + '
        f'({K1[0][1]/m_c})*x1 + ({K1[0][2]/m_c})*x2 + '
        f'({-g_lin})*x0'
    )
    f4 = (
        f'({n**2 + K1[1][1]/m_c})*x1 + '
        f'({K1[1][2]/m_c - 2*n})*x2 + '
        f'({K1[1][0]/m_c})*x0 + ({K1[1][3]/m_c})*x3 + '
        f'({-g_lin})*x1'
    )
    return [f1, f2, f3, f4]

write('SPRE22',
    title='SPRE22 (Approaching mode)',
    n_states=4,
    x_syms=['x0', 'x1', 'x2', 'x3'],
    dynamics=_spre22_dyn(),
    p_syms=[], p_values={},
    L_initial=[-925.0, -425.0, 0.0, 0.0],
    U_initial=[-875.0, -375.0, 5.0, 5.0],
    L_space=[-1000.0, -500.0, -10.0, -10.0],
    U_space=[   50.0,   50.0,  10.0,  10.0],
    L_unsafe=[
        [-1000.0, -500.0,   5.0, -10.0],
        [-1000.0, -500.0, -10.0, -10.0],
        [-1000.0, -500.0, -10.0,   5.0],
        [-1000.0, -500.0, -10.0, -10.0],
    ],
    U_unsafe=[
        [50.0, 50.0, 10.0, 10.0],
        [50.0, 50.0, -5.0, 10.0],
        [50.0, 50.0, 10.0, 10.0],
        [50.0, 50.0, 10.0, -5.0],
    ],
    projections=[
        [0, 1, 'x [m]', 'y [m]'],
        [2, 3, 'v_x', 'v_y'],
        [0, 2, 'x', 'v_x'],
        [1, 3, 'y', 'v_y'],
    ],
)


# ----------------------------------------------------------------------
# TRAF22 -- v2 robust, 4-D state + 1 sinc parameter
# ----------------------------------------------------------------------
psi_max = 0.5
sinc_lo = math.sin(psi_max) / psi_max
write('TRAF22',
    title='TRAF22 (v2 robust SOS)',
    n_states=4,
    x_syms=['x0', 'x1', 'x2', 'x3'],
    dynamics=[
        '0',
        'x2/2.578 * x0',
        '0',
        'x2 * p0 * x1',
    ],
    p_syms=['p0'], p_values={'p0': (sinc_lo + 1.0) / 2.0},
    L_initial=[-0.04, -0.04, 4.9, -0.1],
    U_initial=[ 0.04,  0.04, 5.1,  0.1],
    L_space=[-0.4, -0.5, 4.0, -2.0],
    U_space=[ 0.4,  0.5, 6.0,  2.0],
    L_unsafe=[
        [-0.4, -0.5, 4.0,  1.5],
        [-0.4, -0.5, 4.0, -2.0],
    ],
    U_unsafe=[
        [0.4, 0.5, 6.0,  2.0],
        [0.4, 0.5, 6.0, -1.5],
    ],
    projections=[
        [0, 3, '\\delta', 's_y'],
        [1, 3, '\\psi',   's_y'],
        [2, 3, 'v',       's_y'],
    ],
)


print('done.')
