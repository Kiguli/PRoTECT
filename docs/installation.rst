Installation
============

Requirements
------------

- **Python 3.10** (recommended; latest versions may not work)
- `Mosek <https://www.mosek.com/>`_ solver license (free for academics) — optional, CVXOPT is the default fallback

Install from source
-------------------

Clone the repository and install dependencies:

.. code-block:: bash

   git clone https://github.com/Kiguli/PRoTECT.git
   cd PRoTECT
   pip install -r requirements.txt

Ubuntu quick install
--------------------

On Ubuntu 22.04, an automated script is provided:

.. code-block:: bash

   cd ~/PRoTECT/bash-scripts
   ./install_ubuntu22_PRoTECT.sh

This installs all prerequisites and configures the ``PYTHONPATH``.

Docker
------

PRoTECT can also be run via Docker (API scripts only, no GUI):

.. code-block:: bash

   docker build -t protect .
   docker run --rm -it --name protect protect

To add a Mosek license to the container:

.. code-block:: bash

   docker cp <license-file> protect:<target-path>

Mosek license
-------------

If you choose to use the Mosek solver (faster than CVXOPT), obtain a free academic
license from `mosek.com <https://www.mosek.com/license/request/?i=acp>`_.

Setting PYTHONPATH
------------------

To use PRoTECT from example scripts outside the repository, add it to your path:

.. code-block:: bash

   export PYTHONPATH=$PYTHONPATH:/path/to/PRoTECT

To make this permanent, append the line to ``~/.profile`` (or ``~/.zshrc`` on macOS)
and restart your shell.

Verify the installation
-----------------------

Launch the GUI:

.. code-block:: bash

   python3 main.py

Or run an example script:

.. code-block:: bash

   cd ex/benchmarks-deterministic/PRoTECT-versions
   python3 ex2_jet_engine_ct_DS.py
