Pressure driven stokes flow
----------------------------------

This demo show how to run a pressure-driven Hagen-Poiseuille flow using the stokes flow solver. The inputs are largely similar to :doc:`stokes ldc <app_stokes_ldc>`. Here we will hilight the key differences

=============================
Running the demo
=============================

This demo use the ``rect.h5`` mesh. Please follow the instructions on creating the ``h5`` mesh from :doc:`meshing <meshing>`. Once the mesh is created, navigate to ``apps/`` and simply run

.. code::

    python3 incompressible_flow.py inputs/stokes_pressure_driven.inp


=============================
Input descriptions
=============================

The following are key differences in the input file from :doc:`stokes_ldc <app_stokes_ldc>`

Boundary conditions here are pressure driven. This is specified through the Neumann condition described in :doc:`stokes physics <stokes_flow>`. Here we define the inlet pressure to 10 and outlet pressure to 1. The top and bottom walls are considered no-slip. Note that this is a Neumann, thus a weak imposition of the pressure condition

.. code::

    # Boundary conditions
    inlet pressure = 10.0
    outlet pressure = 1.0
    BC1 = (1, pressure, face, inlet pressure)
    BC2 = (2, wall, no slip)
    BC3 = (3, pressure, face, outlet pressure)
    BC4 = (4, wall, no slip)

Finally, this demo demonstrate how to control the linear algebra solver in PETSc. Here we use the ``gmres`` solver with the incomplete lu preconditioner. In the current stage, the controls are limited to basic krylov solver inputs for petsc

.. code::

    # Linear solver
    solver type = gmres
    pc type = ilu
    ksp relative tolerance = 1e-8
    ksp absolute tolerance = 1e-10
    ksp maximum iterations = 1000
    ksp monitor convergence = true


