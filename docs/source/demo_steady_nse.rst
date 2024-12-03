Steady state Navier-Stokes solver
===================================================

This demo code demonstrates how to solve a steady state incompressible Navier-Stokes equation using the canonical lid driven cavity problem. The full source code can be found in **demo/user_defined/documented/steady_navier_stokes**

Problem definition
--------------------

This problem is the steady state incompressible Navier-Stokes problem in a unit square domain.

Strong form

.. math::

    \rho \textbf{u} \cdot \nabla \textbf{u} = \boldsymbol{\sigma} + \textbf{b} \\

    \nabla \cdot \textbf{u} = 0

Where the stress term is

.. math::

    \boldsymbol{\sigma} = -p\textbf{I} + \mu\left(\nabla \textbf{u} + \nabla \textbf{u}^T \right)

The boundary conditions are

.. math::

    \textbf{u}(x=0) = \textbf{u}(x=1) = \textbf{u}(y=0) = \textbf{0} \\

    \textbf{u}(y=1) = (1, 0) \\

    p(x=0,y=0) = 0


Implementation
-----------------

First we import the relevant modules and create the mesh

.. code-block:: python

    # Build mesh
    ne = 64
    mesh = RectMesh(0, 0, 1, 1, 1/ne)


Next, we define the incompressible Navier Stokes solver using the ``SteadyIncompressibleNavierStokes`` physics. Here we will use the *unstable* elements combination Q1P1 and we will add stabilization.

.. code-block:: python

    # Define nse equation
    nse = SteadyIncompressibleNavierStokes(mesh)
    nse.set_element('CG', 1, 'CG', 1)
    nse.build_function_space()

Next the problem parameters are defined and the weak formulation is set. We call ``add_stab()`` to add the stabilization terms. In this implementation, we include both the SUPG and PSPG stabilization. Please see :doc:`Steady Navier Stokes physics <navier_stokes>` for more detail

.. code-block:: python

    # Set parameters
    Re = 100
    mu = 1/Re
    rho = 1
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)

    # Set weak form
    nse.set_weak_form()
    nse.add_stab()


Next we set the boundary conditions. Note that this is similar to the :doc:`Stokes flow demo <demo_stokes_external_force>`

.. code-block:: python

    # Boundary condition
    zero_v = fe.Constant( (0,0) )
    zero = fe.Constant(0)
    u_bcs = {
            1: {'type': 'dirichlet', 'value': zero_v},
            2: {'type': 'dirichlet', 'value': zero_v},
            3: {'type': 'dirichlet', 'value': zero_v},
            4: {'type': 'dirichlet', 'value': fe.Constant((1, 0))},
            }
    p_bcs = {'point_0': {'type': 'dirichlet', 'value': fe.Constant(0), 'x': (0, 0)}}
    bc_dict = {'u': u_bcs,
               'p': p_bcs}
    nse.set_bcs(bc_dict)

Finally we build the writer to write as paraview readable format and solve the problem. This will generate a directory name ``output/`` with the flow and pressure data to be read using `paraview <https://www.paraview.org/>`_.

.. code-block:: python

    # Set output writer
    nse.set_writer("output", "pvd")

    # Solve and write result
    solver = PhysicsSolver(nse)
    solver.solve()
    nse.write()


