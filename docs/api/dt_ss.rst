Discrete-Time Stochastic (dt-SS)
=================================

Functions for computing barrier certificates for discrete-time stochastic
systems over a finite time horizon with noise.

``dt_SS``
---------

.. code-block:: python

   from src.functions.dt_SS import dt_SS

   result = dt_SS(b_degree, dim, L_initial, U_initial, L_unsafe, U_unsafe,
                  L_space, U_space, x, varsigma, f, t, noise_type="normal",
                  optimize=False, solver="mosek", confidence=None, gam=None,
                  lam=None, c_val=None, mean=None, sigma=None, rate=None,
                  a=None, b=None, l_degree=None)

**Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``b_degree``
     - int
     - Degree of the barrier polynomial.
   * - ``dim``
     - int
     - Dimension of the state space.
   * - ``L_initial``
     - np.ndarray
     - Lower bounds of the initial set. Shape: ``(dim,)``.
   * - ``U_initial``
     - np.ndarray
     - Upper bounds of the initial set. Shape: ``(dim,)``.
   * - ``L_unsafe``
     - np.ndarray
     - Lower bounds of unsafe set(s). Shape: ``(n_regions, dim)``.
   * - ``U_unsafe``
     - np.ndarray
     - Upper bounds of unsafe set(s). Shape: ``(n_regions, dim)``.
   * - ``L_space``
     - np.ndarray
     - Lower bounds of the state space. Shape: ``(dim,)``.
   * - ``U_space``
     - np.ndarray
     - Upper bounds of the state space. Shape: ``(dim,)``.
   * - ``x``
     - tuple
     - SymPy symbols for state variables.
   * - ``varsigma``
     - tuple
     - SymPy symbols for noise variables.
   * - ``f``
     - np.ndarray
     - Array of SymPy expressions defining the dynamics.
   * - ``t``
     - int
     - Finite time horizon.
   * - ``noise_type``
     - str
     - Type of noise: ``"normal"``, ``"exponential"``, or ``"uniform"``.
   * - ``optimize``
     - bool
     - If ``True``, optimize the barrier certificate (requires ``gam`` and ``c_val``). Default: ``False``.
   * - ``solver``
     - str
     - ``"mosek"`` (default) or ``"cvxopt"``.
   * - ``confidence``
     - float or None
     - Minimal confidence threshold for the feasible solution.
   * - ``gam``
     - float or None
     - Fixed value for :math:`\gamma`.
   * - ``lam``
     - float or None
     - Fixed value for :math:`\lambda`.
   * - ``c_val``
     - float or None
     - Value for :math:`c` (multiplied by time horizon in the confidence bound).
   * - ``mean``
     - list or None
     - Mean values for normal noise (assumed zero if not provided).
   * - ``sigma``
     - list or None
     - Standard deviations for normal noise per dimension.
   * - ``rate``
     - list or None
     - Rate parameter(s) for exponential noise.
   * - ``a``
     - list or None
     - Lower bounds for uniform noise.
   * - ``b``
     - list or None
     - Upper bounds for uniform noise.
   * - ``l_degree``
     - int or None
     - Degree of Lagrangian multipliers. Defaults to ``b_degree``.

**Noise-specific parameters:**

- **Normal:** provide ``sigma`` (and optionally ``mean``)
- **Exponential:** provide ``rate``
- **Uniform:** provide ``a`` and ``b``

**Returns:** ``dict`` with barrier certificate details, or ``None`` if infeasible.

----

``parallel_dt_SS``
------------------

.. code-block:: python

   from src.functions.parallel_dt_SS import parallel_dt_SS

   result = parallel_dt_SS(b_degree, dim, L_initial, U_initial, L_unsafe,
                           U_unsafe, L_space, U_space, x, varsigma, f, t,
                           noise_type="normal", optimize=False, solver="mosek",
                           confidence=None, gam=None, lam=None, c_val=None,
                           mean=None, sigma=None, rate=None, a=None, b=None,
                           l_degree=None)

Searches barrier degrees 2, 4, ..., ``b_degree`` in parallel. All parameters
are identical to ``dt_SS`` except ``b_degree`` is the **maximum** degree.

.. note::

   Must be called inside ``if __name__ == '__main__':`` due to Python multiprocessing.

**Returns:** The result ``dict`` from the first successful degree, or ``None``.
