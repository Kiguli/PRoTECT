API Reference
=============

PRoTECT provides barrier certificate computation functions for four classes of
dynamical systems, each with a single-degree and parallelized variant.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Discrete-Time Deterministic (dt-DS)
      :link: dt_ds
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      ``dt_DS`` and ``parallel_dt_DS`` --- infinite time horizon,
      no stochastic noise.

   .. grid-item-card:: Discrete-Time Stochastic (dt-SS)
      :link: dt_ss
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      ``dt_SS`` and ``parallel_dt_SS`` --- finite time horizon
      with Normal, Exponential, or Uniform noise.

   .. grid-item-card:: Continuous-Time Deterministic (ct-DS)
      :link: ct_ds
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      ``ct_DS`` and ``parallel_ct_DS`` --- infinite time horizon,
      Lie derivative conditions.

   .. grid-item-card:: Continuous-Time Stochastic (ct-SS)
      :link: ct_ss
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      ``ct_SS`` and ``parallel_ct_SS`` --- finite time horizon
      with Brownian and Poisson noise.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Utilities
      :link: utilities
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Helper functions and enumerations: ``generate_polynomial``,
      ``SystemMode``, ``NoiseType``.

.. toctree::
   :hidden:

   dt_ds
   dt_ss
   ct_ds
   ct_ss
   utilities
