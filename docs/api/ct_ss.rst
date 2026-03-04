Continuous-Time Stochastic (ct-SS)
===================================

Functions for computing barrier certificates for continuous-time stochastic
systems over a finite time horizon with Brownian and Poisson noise.

``ct_SS``
---------

.. code-block:: python

   from src.functions.ct_SS import ct_SS

   result = ct_SS(b_degree, dim, L_initial, U_initial, L_unsafe, U_unsafe,
                  L_space, U_space, x, f, delta, rho, p_rate, t,
                  optimize=False, solver="mosek", confidence=None, gam=None,
                  lam=None, c_val=None, l_degree=None)

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
   * - ``f``
     - np.ndarray
     - Array of SymPy expressions defining the drift dynamics :math:`\dot{x} = f(x)`.
   * - ``delta``
     - np.ndarray
     - Diffusion coefficient(s) for Brownian motion. Shape: ``(dim,)``.
   * - ``rho``
     - np.ndarray
     - Reset map for Poisson jumps. Shape: ``(dim,)``.
   * - ``p_rate``
     - np.ndarray
     - Poisson rate values. Shape: ``(dim,)``.
   * - ``t``
     - int
     - Finite time horizon.
   * - ``optimize``
     - bool
     - If ``True``, optimize the barrier certificate. Default: ``False``.
   * - ``solver``
     - str
     - ``"mosek"`` (default) or ``"cvxopt"``.
   * - ``confidence``
     - float or None
     - Minimal confidence threshold.
   * - ``gam``
     - float or None
     - Fixed value for :math:`\gamma`.
   * - ``lam``
     - float or None
     - Fixed value for :math:`\lambda`.
   * - ``c_val``
     - float or None
     - Value for :math:`c` (multiplied by time horizon in the confidence bound).
   * - ``l_degree``
     - int or None
     - Degree of Lagrangian multipliers. Defaults to ``b_degree``.

**Returns:** ``dict`` with barrier certificate details, or ``None`` if infeasible.

----

``parallel_ct_SS``
------------------

.. code-block:: python

   from src.functions.parallel_ct_SS import parallel_ct_SS

   result = parallel_ct_SS(b_degree, dim, L_initial, U_initial, L_unsafe,
                           U_unsafe, L_space, U_space, x, f, delta, rho,
                           p_rate, t, optimize=False, solver="mosek",
                           confidence=None, gam=None, lam=None, c_val=None,
                           l_degree=None)

Searches barrier degrees 2, 4, ..., ``b_degree`` in parallel. All parameters
are identical to ``ct_SS`` except ``b_degree`` is the **maximum** degree.

.. note::

   Must be called inside ``if __name__ == '__main__':`` due to Python multiprocessing.

**Returns:** The result ``dict`` from the first successful degree, or ``None``.
