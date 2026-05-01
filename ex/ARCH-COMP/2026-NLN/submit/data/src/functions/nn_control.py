"""
Neural-network-controlled closed-loop verification (PRoTECT v2 feature 10).

Bridges PRoTECT into the AINNCS-style problem space. For a closed loop
``x' = f(x, pi(x))`` with ``pi`` a feed-forward neural network, ReLU
networks have a finite piecewise-affine (PWA) representation: the input
domain partitions into polyhedral cells, and pi is affine on each cell.

We verify safety by:

  1. Enumerating (or sampling) the active PWA cells reachable from the
     initial set.
  2. Per cell ``C_k = {x : H_k x <= h_k}`` with ``pi(x) = W_k x + b_k``,
     instantiate a per-cell ct_DS programme on
         x' = f(x, W_k x + b_k)
     restricted to the polytope cell ``C_k`` (using the polytope
     inequality form from `sets.py`).
  3. Couple cells across boundaries with inclusion-style multipliers
     (similar to the hybrid-system reset map but with the boundary
     hyperplane as the "guard").

For ReLU networks the PWA partition can have exponentially many cells
in the worst case; in practice we use abstraction-refinement: start
from a coarse cell partition (few cells) and refine where the SOS
programme returns infeasible.

This module exposes the cell-extraction API for ReLU networks and the
per-cell ct_DS wrapper. The full abstraction-refinement loop is a
larger v2.1 milestone.
"""

import numpy as np
import sympy as sp


def relu_pwa_cells(weights_layers, biases_layers, input_box):
    """
    Return a list of PWA cells for the ReLU network defined by the
    given weights/biases. Each cell is a dict
        {'H': numpy array (m, n),
         'h': numpy array (m,),
         'W': numpy array (n_out, n_in),  # affine map on the cell
         'b': numpy array (n_out,)}

    Stub: implement via vertex enumeration of activation-pattern hyper-
    cubes, using e.g. the algorithm in Hanin/Rolnick (2019) "Deep
    ReLU Networks Have Surprisingly Few Activation Patterns".
    """
    raise NotImplementedError(
        "PWA cell extraction is scaffolded for v2.1. Use SOPRA or "
        "OVERT for the cell enumeration pass."
    )


def per_cell_dynamics(f_template, x, u_syms, W, b):
    """
    Substitute the affine controller ``u = W x + b`` into the
    parametric dynamics ``f_template(x, u)``. Returns a sympy array
    representing the per-cell closed-loop dynamics.
    """
    if W.shape[1] != len(x):
        raise ValueError("W has wrong number of input columns")
    if W.shape[0] != len(u_syms):
        raise ValueError("W has wrong number of output rows")
    u_exprs = [
        sum(sp.sympify(W[i, j]) * x[j] for j in range(W.shape[1]))
        + sp.sympify(b[i])
        for i in range(W.shape[0])
    ]
    subs = dict(zip(u_syms, u_exprs))
    return [sp.sympify(fi).subs(subs) for fi in f_template]
