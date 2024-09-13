.. title:: Installation


============
Installation
============

DIJITSO is normally installed as part of an installation of FEniCS.
If you are using DIJITSO as part of the FEniCS software suite, it
is recommended that you follow the
`installation instructions for FEniCS
<https://fenics.readthedocs.io/en/latest/>`__.

To install DIJITSO itself, read on below for a list of requirements
and installation instructions.


Requirements and dependencies
=============================

DIJITSO requires Python version 2.7 or later and depends on the
following Python packages:

* NumPy

These packages will be automatically installed as part of the
installation of DIJITSO, if not already present on your system.

Additionally, to run tests the following packages are needed

* pytest
* mpi4py (for running tests with mpi)


Installation instructions
=========================

To install DIJITSO, download the source code from the
`DIJITSO Bitbucket repository
<https://bitbucket.org/fenics-project/dijitso>`__,
and run the following command:

.. code-block:: console

    pip install .

To install to a specific location, add the ``--prefix`` flag
to the installation command:

.. code-block:: console

    pip install --prefix=<some directory> .


Environment
===========

Instant's behaviour depened on following environment variables:

 - ``DIJITSO_CACHE_DIR``

   This option overrides the placement of the cache directory.
   By default the cache directory is placed in ``.cache/dijitso``
   either below the home directory or below the prefix
   of the currently active virtualenv or conda environment if any.

 - ``DIJITSO_SYSTEM_CALL_METHOD``

   Choose method for calling external programs (c++ compiler).
   Available values:

       - ``SUBPROCESS``

           Uses pipes. OFED-fork safe on Python 3. Default.

       - ``OS_SYSTEM``

           Uses temporary files. Probably OFED-fork safe.

.. warning:: OFED-fork safe system call method might be required to
             avoid crashes on OFED-based (InfiniBand) clusters!
