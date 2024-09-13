Stokes flow lid driven cavity
----------------------------------

This demo show how to run a lid-driven cavity problem in the stokes flow regime. See :doc:`here <stokes_flow>` for details on the mathematics of Stokes flow. In this demo, the input file is provided in ``app/inputs/stokes_ldc.inp``



=============================
Running the demo
=============================

This demo use the ``unit_square.h5`` mesh. You can find the ``unit_square.geo`` within ``demo/mesh/geo``. First convert the ``*.geo`` script into the ``h5`` script according to :doc:`meshing <meshing>`. Once meshing is complete, run

.. code::

    python3 incompressible_flow.py inputs/stokes_ldc.inp


=============================
Input descriptions
=============================

First you define the path to the mesh file and the appopriate output prefix and type

.. code::

    # Mesh file
    mesh file = ../demo/mesh/h5/unit_square.h5

    # Output directory prefix
    output prefix = stokes_ldc
    output type = pvd


Next, define the physics as the stokes flow physics using the ``stokes`` keyword

.. code::

    # Set flow physics type
    flow physics type = stokes

Next, define the type of finite element. Currently, two options are available, these are ``taylor hood`` and ``linear``. If ``linear`` element is used, pressure stabilization will be applied to the problem. See :doc:`[1] <references>` and :doc:`stokes <stokes_flow>` for details

.. code::

    # Set element type
    element type = linear


Next, we define the physical parameters of the problem, nameply the ``dynamic viscosity`` and the ``density`` of the fluid. Note that even though dynamic viscosity and density are supplied in the input, the stokes flow solver will internally convert them into a kinematic viscosity.

.. code::

    # Physical properties
    dynamic viscosity = 1
    density = 1


Next is the boundary condition. For lid driven cavity, we fix every boundary except the top boundary as a no slip wall. We then supply a constant, or so called "plug", profile for the top wall to drive the flow. The plug profile is marked as the inlet with flow direction xhat=(1,0), and flow velocity equals to 1.0

.. code::

    # Boundary conditions
    lid_velocity = 1.0
    xhat = (1, 0)
    BC1 = (1, wall, no slip)
    BC2 = (2, wall, no slip)
    BC3 = (3, wall, no slip)
    BC4 = (4, inlet, plug, xhat, lid_velocity)

Finally, to set the level of the pressure, we fix the pressure value at the bottom left corner, x0=(0,0) to value p_point = 0. This is done by using the pressure point type boundary condition. Note that the keyword "point0" is simply a name indicating a point-wise boundary condition. 

.. code::

    p_point = 0.0
    x0 = (0,0)
    BC5 = (point0, pressure, point, p_point, x0)


