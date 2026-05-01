"""
Multi-mode hybrid barriers (PRoTECT v2 feature 3a, end-to-end SOS solver).

For a hybrid automaton with discrete locations Q = {q1, ..., qK}, find
per-mode barriers ``{B_q : q in Q}`` plus consistent reset-map
inclusions on every transition. All B_q are searched JOINTLY in a
single SOS programme so that coupling constraints are tight.

SOS conditions per mode q:
    -B_q(x) - sum L_init_q_i * g_init_q_i + gamma                  is SOS    (i)
     B_q(x) - sum L_unsafe_q_j_i * g_unsafe_q_j_i - lambda          is SOS    (ii)
    -<dB_q/dx, f_q(x)> - sum L_inv_q_i * g_inv_q_i                  is SOS    (iii)

Per edge e = (q -> q') with guard g_e and reset R_e:
    -B_{q'}(R_e(x)) - sum L_e_g_i * g_e_i - L_e_src * (-B_q(x))     is SOS    (iv)

The src-side multiplier ``L_e_src * (-B_q(x))`` localises the inclusion
to states where B_q(x) <= 0.

Single-shared gamma / lambda across modes (so the certificate has a
common level structure).
"""

import numpy as np
import sympy as sp

import picos
from SumOfSquares import SOSProblem, poly_variable


class HybridMode:
    def __init__(self, name, x, dynamics, invariant_polys,
                 initial_polys=None, unsafe_regions_polys=None):
        self.name = name
        self.x = list(x)
        self.dynamics = list(dynamics)
        self.invariant_polys = [sp.sympify(g) for g in invariant_polys]
        self.initial_polys = [sp.sympify(g) for g in (initial_polys or [])]
        self.unsafe_regions_polys = [
            [sp.sympify(g) for g in region]
            for region in (unsafe_regions_polys or [])
        ]


class HybridEdge:
    def __init__(self, src_name, dst_name, guard_polys, reset_map):
        self.src_name = src_name
        self.dst_name = dst_name
        self.guard_polys = [sp.sympify(g) for g in guard_polys]
        self.reset_map = reset_map  # callable: list of source-state syms -> list of sympy exprs


def ct_DS_hybrid(
    b_degree,
    modes,
    edges,
    solver='mosek',
    gam=None, lam=None, l_degree=None,
):
    if l_degree is None:
        l_degree = b_degree

    prob = SOSProblem()
    result = {'b_degree': b_degree}

    Barriers = {}
    L_init = {}
    L_unsafe = {}
    L_inv = {}

    try:
        for mode in modes:
            Barriers[mode.name] = poly_variable(
                f'B_{mode.name}', mode.x, b_degree)
            L_init[mode.name] = [
                poly_variable(f'L_init_{mode.name}_{i+1}',
                              mode.x, l_degree)
                for i in range(len(mode.initial_polys))]
            L_unsafe[mode.name] = [
                [poly_variable(f'L_unsafe_{mode.name}_{j}_{i+1}',
                               mode.x, l_degree)
                 for i in range(len(mode.unsafe_regions_polys[j]))]
                for j in range(len(mode.unsafe_regions_polys))
            ]
            L_inv[mode.name] = [
                poly_variable(f'L_inv_{mode.name}_{i+1}',
                              mode.x, l_degree)
                for i in range(len(mode.invariant_polys))
            ]

        gamma = sp.symbols('gamma_h')
        gv = prob.sym_to_var(gamma); prob.add_constraint(gv > 0)
        lambda_ = sp.symbols('lambda_h')
        lv = prob.sym_to_var(lambda_); prob.add_constraint(lv > 0)
        prob.add_constraint(lv - gv > 0)
    except Exception:
        return {'error': 'init failure', 'b_degree': b_degree}

    # Per-mode SOS conditions
    try:
        for mode in modes:
            B = Barriers[mode.name]
            grad = np.array([sp.diff(B, xi) for xi in mode.x])
            Lie = np.sum(grad * np.array(mode.dynamics))

            # (i) initial
            if mode.initial_polys:
                init_lag_sum = sum(Li * gi for Li, gi in
                                   zip(L_init[mode.name], mode.initial_polys))
                prob.add_sos_constraint(-B - init_lag_sum + gamma, mode.x)

            # (ii) unsafe (per region)
            for j, region in enumerate(mode.unsafe_regions_polys):
                lag_sum = sum(Li * gi for Li, gi in
                              zip(L_unsafe[mode.name][j], region))
                prob.add_sos_constraint(B - lag_sum - lambda_, mode.x)

            # (iii) Lie under invariant
            inv_lag_sum = sum(Li * gi for Li, gi in
                              zip(L_inv[mode.name], mode.invariant_polys))
            prob.add_sos_constraint(-Lie - inv_lag_sum, mode.x)

            # all multipliers SOS in mode.x
            for Li in L_init[mode.name]:
                prob.add_sos_constraint(Li, mode.x)
            for region_lag in L_unsafe[mode.name]:
                for Li in region_lag:
                    prob.add_sos_constraint(Li, mode.x)
            for Li in L_inv[mode.name]:
                prob.add_sos_constraint(Li, mode.x)

            prob.add_sos_constraint(B, mode.x)

    except AssertionError:
        return {'error': 'AssertionError per-mode', 'b_degree': b_degree}

    # Per-edge reset-map inclusion conditions (iv)
    try:
        modes_by_name = {m.name: m for m in modes}
        for e_idx, edge in enumerate(edges):
            src_mode = modes_by_name[edge.src_name]
            dst_mode = modes_by_name[edge.dst_name]
            B_src = Barriers[src_mode.name]
            B_dst = Barriers[dst_mode.name]

            # Reset map: dst state values as functions of src state.
            reset_state = edge.reset_map(src_mode.x)
            subs = dict(zip(dst_mode.x, reset_state))
            B_dst_at_reset = sp.sympify(B_dst).subs(subs)

            # Multipliers (in src state):
            L_g = [poly_variable(f'L_e_{e_idx}_g_{i+1}',
                                 src_mode.x, l_degree)
                   for i in range(len(edge.guard_polys))]
            L_src = poly_variable(f'L_e_{e_idx}_src',
                                  src_mode.x, l_degree)

            terms = -B_dst_at_reset
            terms = terms - sum(Li * gi for Li, gi in zip(L_g, edge.guard_polys))
            terms = terms - L_src * (-B_src)

            prob.add_sos_constraint(terms, src_mode.x)
            for Li in L_g:
                prob.add_sos_constraint(Li, src_mode.x)
            prob.add_sos_constraint(L_src, src_mode.x)

    except AssertionError:
        return {'error': 'AssertionError per-edge', 'b_degree': b_degree}

    try:
        prob.solve(solver=solver)
    except picos.modeling.problem.SolutionFailure:
        return {'error': 'picos SolutionFailure', 'b_degree': b_degree}
    except Exception:
        return {'error': 'Solver Exception', 'b_degree': b_degree}

    try:
        result['barriers'] = {m.name: Barriers[m.name] for m in modes}
        result['gamma'] = float(gv)
        result['lambda'] = float(lv)
        return result
    except Exception:
        return {'error': 'reading hybrid result failed',
                'b_degree': b_degree}
