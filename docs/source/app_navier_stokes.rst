Lid driven cavity Navier-Stokes
-------------------------------------

In this demo, we show how to run the steady-state non-linear lid-driven-cavity problem through the incompressible flow solver. Most of the input will be similar to :doc:`app_stokes_ldc`. Here we will hilight the main difference for the non-linear regime. For details on the underlying mathematics, see :doc:`navier_stokes`.



============================
Running the demo
============================

The non-linear Navier Stokes still fall under the ``incompressible_flow.py`` code. Simply call the python script and supply the input file ``app/inputs/navier_stokes_ldc.inp``


=============================
Input descriptions
=============================

Here we choose the dynamic viscosity to be 0.001 which correspond to Reynold number = 1000.

.. code::

    # Physical properties
    dynamic viscosity = 1e-3
    density = 1

The boundary conditions are identical to the stokes flow lid driven cavity case

.. code::

    # Boundary conditions
    lid_velocity = 1.0
    zero = 0.0
    xhat = (1, 0)
    p_point = 0.0
    x0 = (0,0)
    BC1 = (1, wall, no slip)
    BC2 = (2, wall, no slip)
    BC3 = (3, wall, no slip)
    BC4 = (4, inlet, plug, xhat, lid_velocity)
    BC5 = (point0, pressure, point, p_point, x0)

    # Linear algebra solver
    solver type = direct

Here, we use the Newton iteration to solve the non-linear problem arising from Navier Stokes equations. For a strongly non-linear problem, sometime Newton iteration will struggle to converge if the initial guess is far from the trus solution. Here we provide the so-called sub-viscosity step option where will solve the same problem with higher viscosity, and use the solution as an initial guess for our Newton iteration. In this demo, we turn on ``enable sub viscosity`` option, and supply intermediate sub viscosities. The user can supply successive viscosity steps one wish to use as intermediate steps. In this case, the solver will solve the problem with viscosity equals to 1, and use that solution as the initial guess for the problem with viscosity equals to 1e-1, and so on until viscosity equals to 2e-3. Then the solver will use the final solution as the initial guess for the problem with viscosity supplied by the ``dynamic viscosity`` input.

.. code::

    # Steady navier stokes sub-viscosity solver option
    enable sub viscosity = true
    intermediate sub viscosity = (1, 1e-1, 1e-2, 5e-3, 2e-3)

