Utilities
=========

Helper functions and enumerations used across PRoTECT.

``generate_polynomial``
-----------------------

.. code-block:: python

   from src.functions.generate_polynomial import generate_polynomial

   polys = generate_polynomial(variables, lower_bounds, upper_bounds)

Creates the set-defining polynomials :math:`g_i(x) = (x_i - l_i)(u_i - x_i)` used
to represent rectangular regions in the SOS program. A point :math:`x` is inside
the region when all :math:`g_i(x) \geq 0`.

**Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``variables``
     - tuple
     - SymPy symbols for the state variables.
   * - ``lower_bounds``
     - np.ndarray
     - Lower bounds of the region. Shape: ``(dim,)``.
   * - ``upper_bounds``
     - np.ndarray
     - Upper bounds of the region. Shape: ``(dim,)``.

**Returns:** List of SymPy expressions, one per dimension.

----

``SystemMode``
--------------

Enumeration of the four supported system types.

.. code-block:: python

   from src.utils.system_mode import SystemMode

.. list-table::
   :widths: 30 30
   :header-rows: 1

   * - Member
     - Value
   * - ``SystemMode.DT_DS``
     - ``"discrete_deterministic"``
   * - ``SystemMode.DT_SS``
     - ``"discrete_stochastic"``
   * - ``SystemMode.CT_DS``
     - ``"continuous_deterministic"``
   * - ``SystemMode.CT_SS``
     - ``"continuous_stochastic"``

----

``NoiseType``
-------------

Enumeration of noise distributions for stochastic systems.

.. code-block:: python

   from src.utils.noise_type import NoiseType

.. list-table::
   :widths: 30 30
   :header-rows: 1

   * - Member
     - Value
   * - ``NoiseType.normal``
     - ``"Normal"``
   * - ``NoiseType.exponential``
     - ``"Exponential"``
   * - ``NoiseType.uniform``
     - ``"Uniform"``

----

``doublefactorial`` / ``factorial``
-----------------------------------

Mathematical utility functions used internally by the stochastic barrier
computations for computing moments of noise distributions.

.. code-block:: python

   from src.functions.doublefactorial import doublefactorial
   from src.functions.factorial import factorial
