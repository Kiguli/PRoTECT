"""
Sparse SOS via term sparsity / chordal decomposition (PRoTECT v2 feature 8a).

Standard SOS programmes have ``O(C(n+d, d)^2)`` PSD coefficient count,
which becomes the bottleneck for high-dim benchmarks (TSPS25 at 15+
dim). Term-sparsity exploits the fact that a polynomial of small support
sigma admits an SOS decomposition over a much smaller monomial basis
(the "term-sparse" basis), at the cost of one extra round of basis
expansion if the test fails.

This module is a SCAFFOLD: it wires the API for term-sparse SOS but
delegates the heavy work to the TSSOS Julia/MATLAB tool (or its Python
port) when available. Direct Python re-implementation of TSSOS is a
larger v2.1 milestone.

Reference: Wang, Magron, Lasserre, "TSSOS: A Moment-SOS hierarchy that
exploits term sparsity", SIAM J. Optim. 31 (2021).
"""


def term_sparse_basis(poly, x):
    """
    Compute the term-sparse monomial basis for an SOS test on ``poly``.
    Returns a list of sympy monomials.

    Currently a stub that falls back to the dense basis. Replace with a
    chordal-graph-based filter once integration with TSSOS is available.

    See: https://github.com/wangjie212/TSSOS for the reference algorithm.
    """
    raise NotImplementedError(
        "Term-sparse SOS is scaffolded; integrate with TSSOS or implement "
        "the chordal-graph algorithm from Wang/Magron/Lasserre 2021."
    )


def block_sparse_solve(prob, sparsity_pattern):
    """
    Run an SOS programme using a user-provided block-sparsity pattern.
    Stub for v2.1.
    """
    raise NotImplementedError("Block-sparse solve is scaffolded for v2.1.")
