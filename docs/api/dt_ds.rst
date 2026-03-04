Discrete-Time Deterministic (dt-DS)
====================================

Functions for computing barrier certificates for discrete-time deterministic
systems over an infinite time horizon.

``dt_DS``
---------

.. code-block:: python

   from src.functions.dt_DS import dt_DS

   result = dt_DS(b_degree, dim, L_initial, U_initial, L_unsafe, U_unsafe,
                  L_space, U_space, x, f, solver="mosek", gam=None, lam=None,
                  l_degree=None)

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
     - SymPy symbols for state variables, e.g. ``sp.symbols('x1:3')``.
   * - ``f``
     - np.ndarray
     - Array of SymPy expressions defining the dynamics :math:`x_{k+1} = f(x_k)`.
   * - ``solver``
     - str
     - ``"mosek"`` (default) or ``"cvxopt"``.
   * - ``gam``
     - float or None
     - Fixed value for :math:`\gamma`. If ``None``, determined by the solver.
   * - ``lam``
     - float or None
     - Fixed value for :math:`\lambda`. If ``None``, determined by the solver.
   * - ``l_degree``
     - int or None
     - Degree of Lagrangian multipliers. Defaults to ``b_degree``.

**Returns:** ``dict`` with keys ``b_degree``, ``Barrier``, ``gamma``, ``lambda``, ``solver_status``, or ``None`` if infeasible.

----

``parallel_dt_DS``
------------------

.. code-block:: python

   from src.functions.parallel_dt_DS import parallel_dt_DS

   result = parallel_dt_DS(b_degree, dim, L_initial, U_initial, L_unsafe,
                           U_unsafe, L_space, U_space, x, f, solver="mosek",
                           gam=None, lam=None, l_degree=None)

Searches barrier polynomial degrees 2, 4, ..., ``b_degree`` in parallel using
multiprocessing. Parameters are identical to ``dt_DS`` except ``b_degree`` is the
**maximum** degree to search.

.. note::

   Must be called inside ``if __name__ == '__main__':`` due to Python multiprocessing.

**Returns:** The result ``dict`` from the first successful degree, or ``None``.
