Time dependent Navier-Stokes
==================================

In this demo, we demonstrate how to run the time dependent Navier-Stokes equation with a time varying inlet condition. The source code can be found in ``demo/user_defined/unsteady_navier_stokes/demo_unsteady_navier_stokes.py``. This problem solves the 2D flow over a cylinder problem resulting in a vortex shedding pattern. The benchmark problem follows the problem definition in this `benchmark <https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html>`_.


Problem definition
-----------------------------------

This demo solves a 2D flow over a cylinder problem under a time-periodic inlet flow. The flow domain is a rectangular domain spanning the space :math:`(x,y) \in [0,2.2] \times [0,0.41]` with a circular hole of radius :math:`r=0.05` located at :math:`(x,y)=(0.2,0.2)` (see `benchmark <https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html>`_) for viz. The strong form of the problem is the incompressible Navier-Stokes equation

.. math::

    \rho \frac{\partial \textbf{u}}{\partial t} + \rho \textbf{u} \cdot \nabla \textbf{u} = \nabla \cdot \boldsymbol{\sigma} + \textbf{b}

and

.. math::

    \nabla \cdot \textbf{u} = 0

The `benchmark <https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html>`_ case defines the problem using kinematic viscosity. To match this definition, we will set the density :math:`\rho=1` and the dynamic viscosity :math:`\mu=0.001` in our problem.

The boundary condition for the inlet is:

.. math::

    \textbf{u}(x=0) = \left( U(t)\frac{4y(0.41-y)}{0.41^2}, 0 \right)

where

.. math::

    U(t) = 1.5sin(\pi t/8)

and the top and bottom walls:

.. math::

    \textbf{u}(y=0) = \textbf{u}(y=0.41) = \textbf{0}

Finally impose a "do nothing" condition on the outlet wall.


Generating a mesh
--------------------

This problem requires a more complex mesh domain which is not available in the :doc:`basic meshing module <mesh>`. In this demo we will use `GMSH <https://gmsh.info/>`_ to generate the mesh file. The mesh we will use is the ``foc.geo`` which can be found in ``demo/mesh/geo/foc.geo``. You will need to use the ``geo2h5`` script and the ``clean_mesh_file`` script provided in FLATiron to create the ``*.h5`` mesh format used within the simulation. Please see the instructions in :doc:`GMSH interface instructions <meshing>` on how to use ``geo2h5`` and ``clean_mesh_file``.


Implementation
--------------------

First we load the appopriate libraries and initialize the mesh. Note the relative path from the demo script to the generated ``foc.h5`` mesh file through the GMSH interface

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import copy
    from flatiron_tk.physics import IncompressibleNavierStokes
    from flatiron_tk.io import h5_mod
    from flatiron_tk.mesh import Mesh
    from flatiron_tk.solver import PhysicsSolver
    import fenics as fe

    # Build mesh
    mesh = Mesh(mesh_file='../../../mesh/h5/foc.h5')

Next we build the time dependent incompressible Navier-Stokes solver. Here we will use the unstable Q1P1 elements with SUPG and PSPG stabilization. We use a short-hand for stabilization where we set ``stab=True`` within the ``set_weak_form`` method directly.

.. code:: python

    # Build the nse physics
    nse = IncompressibleNavierStokes(mesh)
    nse.set_element('CG', 1, 'CG', 1)
    nse.build_function_space()

    # Set parameters
    dt = 0.00625
    mu = 0.001
    rho = 1
    nse.set_time_step_size(dt)
    nse.set_mid_point_theta(0.5)
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)

    # Set weak form
    nse.set_weak_form(stab=True)

Next we define the boundary conditions. Here the inlet is a FEniCS ``Expression`` which has the variable ``t``. This value ``t`` will be updated during the time loop

.. code:: python

    # Boundary condition
    U = 1.5
    D = 0.1
    H = 4.1*D
    inlet = fe.Expression(("4*1.5*sin(pi*t/8)*x[1]*(H-x[1])/(H*H)","0"), U=U, H=H, t=0, degree=2)
    zero_v = fe.Constant( (0,0) )
    zero = fe.Constant(0)
    u_bcs = {
            1: {'type': 'dirichlet', 'value': inlet},
            2: {'type': 'dirichlet', 'value': zero_v},
            4: {'type': 'dirichlet', 'value': zero_v},
            5: {'type': 'dirichlet', 'value': zero_v}
            }
    p_bcs = {3: {'type': 'dirichlet', 'value': zero}}
    bc_dict = {'u': u_bcs,
               'p': p_bcs}
    nse.set_bcs(bc_dict)


Next we set the solver and writer. Note that we will save the results as a paraview readable formath using the ``pvd`` input in ``set_writer``.

.. code:: python

    # Set output writer
    nse.set_writer("output", "pvd")

    # Set solver
    solver = PhysicsSolver(nse)

Next, we define the post-processing step computing the coefficient of lift and drag which will be plotted directly during the simulation.

.. code:: python

    # Diagnostics
    # n here is pointing in-ward, so we use the negative
    # to get the force the flow applies onto the cylinder
    def CD(u,p):
        n = mesh.facet_normal()
        u_t = fe.inner( u, fe.as_vector((n[1], -n[0])) )
        return fe.assemble( -2/0.1 * (mu/rho * fe.inner( fe.grad(u_t), n ) * n[1] - p * n[0] ) * nse.ds(5) )

    def CL(u,p):
        n = mesh.facet_normal()
        u_t = fe.inner( u, fe.as_vector((n[1], -n[0])) )
        return fe.assemble( 2/0.1 * (mu/rho * fe.inner( fe.grad(u_t), n ) * n[0] + p * n[1]) * nse.ds(5) )


Finally we solve the problem over time. Notice that we set ``inlet.t = t`` at each iteration to update the time-dependent inlet flow value. The function ``fe.assemble()`` found within the ``CD`` and ``CL`` functions integrates the ufl formulation across all processes, therefore all MPI ranks will have the same integrated result. Within the time integration loop, we only plot the coefficient values on MPI rank 0.

.. code:: python

    # Solve
    t = 0
    i = 0
    Fd = []
    Fl = []
    time = []
    fig, ax = plt.subplots(nrows=2)
    rank = mesh.comm.rank
    while t < 8:

        # Update time and time dependent inlet
        t += dt
        inlet.t = t

        # Solve
        solver.solve()
        nse.update_previous_solution()

        if i%10 == 0:
            nse.write()
        (u, p) = nse.solution_function().split(deepcopy=True)

        LIFT = CL(u, p)
        DRAG = CD(u, p)
        Fl.append(LIFT)
        Fd.append(DRAG)
        time.append(copy.deepcopy(t))

        if i%10 == 0 and rank == 0:
            np.save('time.npy', np.array(time))
            np.save('drag.npy', np.array(Fd))
            np.save('lift.npy', np.array(Fl))

            ax[0].plot(np.array(time), np.array(Fd))
            ax[0].set_ylabel('CD')
            ax[0].set_xlim([0, 8])
            ax[0].set_ylim([-0.5, 3])
            ax[0].grid(True)
            ax[1].plot(np.array(time), np.array(Fl))
            ax[1].set_ylabel('CL')
            ax[1].set_xlabel('Time')
            ax[1].set_xlim([0, 8])
            ax[1].set_ylim([-0.5, 0.5])
            ax[1].grid(True)

            plt.pause(0.0001)
            plt.savefig("CLCD.png")
            ax[0].cla()
            ax[1].cla()

        if rank == 0: 
            print('-'*50)
            print("Writing output at time step: %d"%i)
            print('-'*50)

        i += 1

