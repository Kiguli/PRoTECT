Quick Start
===========

This guide walks through a minimal example: verifying a 2D jet engine system
using a continuous-time deterministic barrier certificate.

1. Define the system
--------------------

.. code-block:: python

   import sympy as sp
   import numpy as np
   from src.functions.ct_DS import ct_DS
   from src.functions.parallel_ct_DS import parallel_ct_DS

   dim = 2  # state-space dimension

   # Initial set bounds
   L_initial = np.array([0.1, 0.1])
   U_initial = np.array([0.5, 0.5])

   # Unsafe set bounds (one or more regions)
   L_unsafe = np.array([[0.7, 0.7]])
   U_unsafe = np.array([[1.0, 1.0]])

   # State-space bounds
   L_space = np.array([0.1, 0.1])
   U_space = np.array([1.0, 1.0])

   # Symbolic variables
   x = sp.symbols('x1:3')  # creates x1, x2

   # Dynamics
   f1 = -x[1] - 3/2 * x[0]**2 - 1/2 * x[0]**3
   f2 = x[0]
   f = np.array([f1, f2])

2. Compute a barrier certificate (single degree)
-------------------------------------------------

.. code-block:: python

   result = ct_DS(
       b_degree=2,
       dim=dim,
       L_initial=L_initial,
       U_initial=U_initial,
       L_unsafe=L_unsafe,
       U_unsafe=U_unsafe,
       L_space=L_space,
       U_space=U_space,
       x=x,
       f=f,
       solver="mosek",
   )

   if result is None:
       print("No feasible barrier found at degree 2.")
   else:
       print("Barrier certificate:", result)

3. Parallelized search across degrees
--------------------------------------

Search barrier degrees 2, 4, and 6 in parallel:

.. code-block:: python

   if __name__ == '__main__':
       result = parallel_ct_DS(
           b_degree=6,  # max degree to search
           dim=dim,
           L_initial=L_initial,
           U_initial=U_initial,
           L_unsafe=L_unsafe,
           U_unsafe=U_unsafe,
           L_space=L_space,
           U_space=U_space,
           x=x,
           f=f,
           solver="mosek",
       )
       print(result)

.. note::

   The parallel functions must be called inside ``if __name__ == '__main__':``
   due to Python's multiprocessing requirements.

4. Understanding the result
---------------------------

When a feasible barrier certificate is found, PRoTECT returns a dictionary containing:

- ``b_degree``: the degree of the barrier polynomial
- ``Barrier``: the symbolic barrier certificate expression
- ``gamma``, ``lambda``: the optimization parameters
- ``solver_status``: status from the SOS solver

If no feasible solution exists at any searched degree, ``None`` is returned.

Using the GUI
-------------

For interactive exploration, launch the graphical interface:

.. code-block:: bash

   python3 main.py

You can import pre-configured examples from the ``ex/GUI_config_files/`` folder
using the **Import Config** button.

.. tip::

   Tutorial videos are available on
   `YouTube <https://www.youtube.com/playlist?list=PL50OJg3FHS4ctLItbuyT5Hqqn6HQzJ_g->`_.

Stochastic systems
------------------

For stochastic systems (dt-SS, ct-SS), additional parameters are required:

.. code-block:: python

   from src.functions.dt_SS import dt_SS

   # Additional symbolic noise variables
   varsigma = sp.symbols('varsigma1:3')

   result = dt_SS(
       b_degree=4,
       dim=dim,
       L_initial=L_initial,
       U_initial=U_initial,
       L_unsafe=L_unsafe,
       U_unsafe=U_unsafe,
       L_space=L_space,
       U_space=U_space,
       x=x,
       varsigma=varsigma,
       f=f,
       t=10,                  # time horizon
       noise_type="normal",   # "normal", "exponential", or "uniform"
       solver="mosek",
       sigma=[0.01, 0.01],    # noise std deviations
   )

See :doc:`api/dt_ss` and :doc:`api/ct_ss` for the full parameter reference.
