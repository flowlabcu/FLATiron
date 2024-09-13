Vortex street demo
-------------------------------------

In this demo, we solve the classic vortex shedding problem using the transient version of the incompressible flow solver. For the details on the background mathematics, please refer to :doc:`transient_navier_stokes`.


============================
Running the demo
============================

This demo use the flow-over-cylinder mesh contained in ``mesh/geo/foc.geo``. Please generate the ``h5`` version of the mesh following the instructions in :doc:`meshing <meshing>`. Once the mesh is created, move the working directory to ``apps/`` and run

.. code::

    python3 incompressible_flow.py inputs/transient_navier_stokes.inp


=============================
Input descriptions
=============================

The inputs will be mostly identical to the :doc:`app_navier_stokes` demo. We will only hilight the main differences within this specific demo.

Firstly, specify the physics to ``navier stokes``.

.. code::

    # Steady navier stokes
    flow physics type = navier stokes


Next, time integration parametrs must be set. Here you must specify the time step size, the total simulation time span, and the output saving frequency

.. code::

    # Time dependent variables
    time step size = 0.00625
    time span = 0.5
    save every = 10

Next is the boundary condition. Here we specify a parabolic profile using the ``inlet`` keyword on boundary face ``1``. The ``inlet`` boundary condition keyword take in 5 additional inputs, these are the ``profile type``, ``flow direction``, ``centerline speed``, ``face center``, and ``face radius``.

The ``flow profile`` in this case is supplied as ``parabolic``.

The ``flow direction`` here is supplied as the keyword ``wall``. If the keyward ``wall`` is used, the solver will assume that the inlet face is `flat`, i.e., every elements on the face has the same normal direction. feFlow will, then, assume that the flow direction be the **inward** pointing normal to the associated boundary. In this case, one can modify the ``flow direciton`` variable to ``flow direciton = (1,0)`` and you will get the same result

The ``centerline speed`` is indicated as the input variable ``U`` and it is the velocity at the peak of the parabolic profile.

Next, the ``face center`` variable is the location where the flow velocity peaks

Finally, the ``face radius`` variable indicate the distance from ``face center`` where the flow velocity go to 0. Any point on the surface further away from ``face center`` will have zero velocity.

.. code::

    # Boundary conditions
    zero = 0.0
    flow direction = wall
    zero = 0.0
    U = 1.5
    xc = (0, 0.205)
    r = 0.205
    BC1 = (1, inlet, parabolic, flow direction, U, xc, r)
    BC2 = (2, wall, no slip)
    BC3 = (3, pressure, face dirichlet, zero)
    BC4 = (4, wall, no slip)
    BC5 = (5, wall, no slip)

Finally, we set the linear algebra solver. Please note that here we do not have the sub-viscosity option. This is because the previous time step solution will be used as an initial guess which should be a sufficiently close initial guess given a sufficiently smalle time step size

.. code::

    # Linear solver
    solver type = direct


