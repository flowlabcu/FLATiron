Time dependent Navier-Stokes
-----------------------------------

In this demo, we demonstrate how to run the time dependent Navier-Stokes equation with a time varying inlet condition. The source code can be found in ``demo/user_defined/unsteady_navier_stokes/demo_unsteady_navier_stokes.py``.

This is a benchmark problem following the defition in https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html


===================
Code description
===================

First we load the appopriate libraries and set the constants

.. code:: python

    from flatiron_tk.physics import IncompressibleNavierStokes
    from flatiron_tk.io import h5_mod
    from flatiron_tk.mesh import Mesh
    from flatiron_tk.solver import PhysicsSolver
    import fenics as fe

    # Constants
    dt = 0.00625
    mu = 0.001
    rho = 1

Next we load the flow over cylinder mesh

.. code:: python

    mesh = Mesh(mesh_file='../../mesh/h5/foc.h5')
    nse = IncompressibleNavierStokes(mesh)
    nse.set_element('CG', 1, 'CG', 1)
    nse.build_function_space()

Next we set the time dependent variables and build the weak form

.. code:: python

    # Set parameters
    nse.set_time_step_size(dt)
    nse.set_mid_point_theta(0.5)
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)

    # Set weak form
    nse.set_weak_form(stab=True)

Next we define the boundary conditions. Here the inlet ``Expression`` has the variable ``t``. This value ``t`` will be changed during the time loop

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

Next we set the solver and writer

.. code:: python

    # Set output writer
    nse.set_writer("output", "pvd")

    # Set solver
    la_solver = fe.LUSolver()
    solver = PhysicsSolver(nse, la_solver)

Here we define the coefficient of lift and drag for diagnostics

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


Finally we solve the problem over time. Notice that we set ``inlet.t = t`` at each iteration to update the time-dependent inlet flow value

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

        if i%10 == 0:
            if rank == 0:
                np.save('time.npy', np.array(time))
                np.save('drag.npy', np.array(Fd))
                np.save('lift.npy', np.array(Fl))

                ax[0].plot(np.array(time), np.array(Fd))
                ax[0].set_title('Drag')
                ax[0].set_xlim([0, 8])
                ax[0].set_ylim([-0.5, 3])
                ax[0].grid(True)
                ax[1].plot(np.array(time), np.array(Fl))

                ax[1].set_title('Lift')
                ax[1].set_xlim([0, 8])
                ax[1].set_ylim([-0.5, 0.5])
                ax[1].grid(True)

                plt.pause(0.0001)
                plt.savefig("CLCD.png")
                ax[0].cla()
                ax[1].cla()

        if rank == 0: print('---------------------------------\nWriting output at timestep: %d.\n---------------------------------' %(i))
        i += 1





