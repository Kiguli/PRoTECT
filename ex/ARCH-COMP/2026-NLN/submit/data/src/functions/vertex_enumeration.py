"""
Vertex enumeration for parametric uncertainty (PRoTECT v2 feature 7b).

Alternative to the robust-Positivstellensatz encoding (`ct_DS_robust`):
solve N independent SOS programmes at the vertices of the parameter
polytope and intersect the resulting barrier level sets. The intersection
is a sound certificate valid for every parameter in the convex hull of
the vertices, provided the dynamics are AFFINE in the parameter.

Trade-offs vs `ct_DS_robust`:
    - + Each per-vertex programme is in n-D state only (no parameter
        dimension lift), so each programme is small.
    - + Embarrassingly parallel: N independent solves.
    - - The intersection of level sets generally produces a non-convex
        certificate; only useful if downstream consumers accept that.
    - - Requires the dynamics to be affine in p (otherwise the
        convex-hull conclusion is unsound). Use `ct_DS_robust` for
        non-affine parameter dependence.

Usage
-----
>>> from src.functions.vertex_enumeration import enumerate_vertices, run_at_vertices
>>> verts = enumerate_vertices([(0.5, 1.5), (1.0, 3.0)])
>>> # verts is the 4-element list [(0.5, 1.0), (0.5, 3.0), (1.5, 1.0), (1.5, 3.0)]
>>> results = run_at_vertices(solve_callable, verts)
>>> # results[k] is the per-vertex result dict
"""

import itertools


def enumerate_vertices(parameter_ranges):
    """
    Cartesian product of the per-parameter (lo, hi) endpoints.

    Parameters
    ----------
    parameter_ranges : sequence of (lo, hi) tuples, length m

    Returns
    -------
    list of tuples of length m, total length 2**m
    """
    endpoint_pairs = [(lo, hi) for (lo, hi) in parameter_ranges]
    return list(itertools.product(*endpoint_pairs))


def run_at_vertices(solve_fn, vertices, **shared_kwargs):
    """
    Call ``solve_fn(p_values, **shared_kwargs)`` once per vertex and
    return a list of results.

    The caller is responsible for building solve_fn appropriately --
    typically a closure over `ct_DS` that pins the parameter symbols to
    the vertex values before solving.
    """
    return [solve_fn(p_values, **shared_kwargs) for p_values in vertices]


def intersect_level_sets(results, gamma_key='gamma'):
    """
    Given a list of per-vertex result dicts (each with a 'barrier' and a
    'gamma'), return the conservative certificate as a dict containing
        - 'barrier_max'  : the max-of-barriers polynomial (i.e., trajectory
                           is safe if max_k B_k(x) >= 0 fails -> use min)
        - 'gamma'        : the worst-case gamma over the vertices.
    Concretely the safety guarantee is "trajectory stays in
    {x : min_k B_k(x) <= gamma_max}", i.e. the *intersection* of the
    individual safe regions.

    Note: this returns the per-vertex barriers and the worst gamma; the
    actual ``min`` operator is applied symbolically by the caller (or
    numerically when plotting / verifying).
    """
    barriers = [r['barrier'] for r in results if 'barrier' in r]
    if len(barriers) == 0:
        return {'error': 'no per-vertex barrier was found'}
    gammas = [r[gamma_key] for r in results if gamma_key in r]
    return {
        'barriers': barriers,
        'gamma_max': max(gammas) if gammas else None,
        'n_vertices': len(barriers),
    }
