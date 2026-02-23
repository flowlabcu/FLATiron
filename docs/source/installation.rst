====================================
Installation
====================================

To install FLATiron, you must have Python 3.8 or higher installed on your system. It is recommended to use a virtual environment to manage dependencies.

Quick Installation Guide
--------------------------------------------

Install the following required libraries before installing FLATiron:

1. **dolfinx version 0.9.0:** *(recommended intallation method: conda)*
    
    You can find instructions for installing `dolfinx` on the `FEniCS project website <https://fenicsproject.org/download/>`__.

2. **ADIOS4DOLFINx version 0.9.4:** *(recommended intallation method: conda)*
    
    You can find instructions for installing `adios4dolfinx` on their `GitHub repository <https://github.com/jorgensd/adios4dolfinx/tree/main>`__.

3. **GMSH:** *(recommended intallation method: pip*)*
    
    You can find instructions for installing `GMSH` on the `GMSH website <https://gmsh.info/#Download>`__.

Once you have installed the required libraries, you can install FLATiron using pip.
Download the source code for FLATiron, navigate into the FLATiron directory and run:

.. code-block:: bash

        pip install .

Detailed Installation Walkthrough
--------------------------------------------

FLATiron is intended to be installed within an environment manager like Anaconda on
UNIX-like systems (macOS, Linux, Ubuntu, etc.). Windows users should install a Linux
subsystem. These instructions assume you are using Anaconda on a UNIX-like system. Here, we 
install FLATiron into the home directory. You may install
FLATiron into any directory on your UNIX-like system.

Starting from the home directory:
1. Create a new conda environment to install FLATiron into:

     .. code-block:: bash

             conda create -n FLATiron-env

2. Activate the new Anaconda environment and confirm that the environment name appears next to your prompt:

     .. code-block:: bash

             conda activate FLATiron-env

3. Install the libmamba solver for faster installation:

     .. code-block:: bash

             conda install conda-libmamba-solver

4. Install `FEniCSx-dolfinx <https://fenicsproject.org/download/>`_, `mpich <https://anaconda.org/channels/anaconda/packages/mpich/overview>`_, and 
`pyvista <https://docs.pyvista.org/getting-started/installation>`_. Notice we are installing **dolfinx v0.9.0** — you must specify the DOLFINx version number during installation. 
We set the solver to libmamba.

     .. code-block:: bash

             conda install -c conda-forge fenics-dolfinx=0.9.0 mpich pyvista --solver=libmamba

5. Install `ADIOS4DOLFINx <https://github.com/jorgensd/adios4dolfinx>`_ using conda. ADIOS4DOLFINx is available on its GitHub page.

     .. code-block:: bash

             conda install -c conda-forge adios4dolfinx-0.9.4 

6. Install `GMSH <https://gmsh.info/#Download>`_ using pip:

     .. code-block:: bash

             pip install gmsh

7. Navigate to the `FLATiron GitHub <https://github.com/flowlabcu/FLATiron#>`_ page. Clone the repository using git (recommended), or download and extract the source archive. Clone the source code from GitHub:

     .. code-block:: bash

             git clone git@github.com:flowlabcu/FLATiron.git

8. Navigate to the top level of the FLATiron source directory:

     .. code-block:: bash

             cd FLATiron

9. Install FLATiron using pip:

     .. code-block:: bash

             pip install .

Congratulations! You have successfully installed FLATiron.