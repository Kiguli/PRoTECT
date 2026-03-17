:html_theme.sidebar_secondary.remove:

.. meta::
   :description: PRoTECT — Parallelized Construction of Safety Barrier Certificates
   :keywords: barrier certificates, safety verification, polynomial systems, SOS optimization

=======
PRoTECT
=======

**Parallelized Construction of Safety Barrier Certificates for Nonlinear Polynomial Systems**

PRoTECT is an open-source tool for verifying safety properties of dynamical systems
through barrier certificate synthesis. It supports four classes of systems ---
discrete-time and continuous-time, deterministic and stochastic --- using
sum-of-squares (SOS) optimization with parallelized search across barrier degrees.

.. grid:: 2 2 4 4
   :gutter: 3

   .. grid-item-card:: :octicon:`zap;1.5em` Parallelized Search
      :class-card: sd-border-0 sd-shadow-sm

      Searches across multiple barrier polynomial degrees in parallel
      using multiprocessing for faster certificate discovery.

   .. grid-item-card:: :octicon:`git-branch;1.5em` 4 System Types
      :class-card: sd-border-0 sd-shadow-sm

      Supports discrete-time and continuous-time systems, both
      deterministic and stochastic (dt-DS, dt-SS, ct-DS, ct-SS).

   .. grid-item-card:: :octicon:`device-desktop;1.5em` GUI + API
      :class-card: sd-border-0 sd-shadow-sm

      Use the PyQt6 graphical interface for interactive exploration
      or call functions directly from Python scripts.

   .. grid-item-card:: :octicon:`verified;1.5em` SOS Optimization
      :class-card: sd-border-0 sd-shadow-sm

      Employs sum-of-squares optimization programs to systematically
      search for polynomial-type barrier certificates.

----

Getting Started
===============

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Install PRoTECT and its dependencies on Ubuntu, macOS, or Docker.

   .. grid-item-card:: Quick Start
      :link: quickstart
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Define a system, compute a barrier certificate, and interpret the results.

   .. grid-item-card:: Examples
      :link: examples
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Browse worked examples with figures: jet engine, Van der Pol, two-room, and more.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Complete reference for all barrier computation functions and utilities.

----

Supported System Types
======================

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - System Type
     - Time
     - Noise
     - Functions
   * - Discrete-Time Deterministic (dt-DS)
     - Discrete
     - None
     - ``dt_DS``, ``parallel_dt_DS``
   * - Discrete-Time Stochastic (dt-SS)
     - Discrete
     - Normal, Exponential, Uniform
     - ``dt_SS``, ``parallel_dt_SS``
   * - Continuous-Time Deterministic (ct-DS)
     - Continuous
     - None
     - ``ct_DS``, ``parallel_ct_DS``
   * - Continuous-Time Stochastic (ct-SS)
     - Continuous
     - Normal, Exponential, Uniform
     - ``ct_SS``, ``parallel_ct_SS``

----

Citing PRoTECT
==============

If you use PRoTECT in your research, please cite:

.. code-block:: bibtex

   @inproceedings{wooding2025ictac,
     title={PRoTECT: Parallelized ConstRuction of SafeTy BarriEr
            Certificates for Nonlinear Polynomial SysTems},
     author={Wooding, Ben and Horbanov, Viacheslav and Lavaei, Abolfazl},
     booktitle={International Colloquium on Theoretical Aspects of Computing},
     pages={448--458},
     year={2025},
     organization={Springer}
   }

   @inproceedings{wooding2025iccps,
     title={Protect: Parallel construction of barrier certificates for safety verification of polynomial systems},
     author={Wooding, Ben and Horbanov, Viacheslav and Lavaei, Abolfazl},
     booktitle={Proceedings of the ACM/IEEE 16th International Conference on Cyber-Physical Systems (with CPS-IoT Week 2025)},
     pages={1--2},
     year={2025}
   }

.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :hidden:
   :caption: API Reference

   api/index
