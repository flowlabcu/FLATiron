======================================================================
Demo: Transient Navier-Stokes
======================================================================

In this demo, we demonstrate how to run the time dependent Navier-Stokes equation with a time varying inlet condition.
The source code can be found in ``demo/demo_transient_navier_stokes/demo_transient_navier_stokes.py``. This problem 
solves the 2D flow over a cylinder problem resulting in a vortex shedding pattern. The benchmark problem follows the problem
definition in this `benchmark <https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html>`_.

This demo solves a 2D flow over a cylinder problem under a time-periodic inlet flow. The flow domain is a rectangular domain 
spanning the space :math:`(x,y) \in [0,2.2] \times [0,0.41]` with a circular hole of radius :math:`r=0.05` 
located at :math:`(x,y)=(0.2,0.2)` (see `benchmark <https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html>`_) for viz. 
The strong form of the problem is the incompressible Navier-Stokes equation:

.. math::

    \rho \frac{\partial \textbf{u}}{\partial t} + \rho \textbf{u} \cdot \nabla \textbf{u} = \nabla \cdot \boldsymbol{\sigma} + \textbf{b}

and

.. math::

    \nabla \cdot \textbf{u} = 0

The `benchmark <https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html>`_ case defines 
the problem using kinematic viscosity. To match this definition, we will set the density :math:`\rho=1` and the dynamic 
viscosity :math:`\mu=0.001` in our problem.

The boundary condition for the inlet is:

.. math::

    \textbf{u}(x=0) = \left( U(t)\frac{4y(0.41-y)}{0.41^2}, 0 \right)

where

.. math::

    U(t) = 1.5sin(\pi t/8)

and the top, bottom, and cylinder walls:

.. math::

    \textbf{u}(y=0) = \textbf{u}(y=0.41) = \textbf{0}

Finally impose a "do nothing" condition on the outlet wall. 

Generating the mesh
--------------------
This problem requires a more complex mesh domain which is not available in the :doc:`basic meshing module <../_modules/module_mesh>`. 
In this demo we will use `GMSH <https://gmsh.info/>`_ to generate the mesh file. The mesh we will use is the ``foc.geo`` which can 
be found in ``demo/mesh/geo/foc.geo``. 

To generate the mesh, navigate to the ``demo/mesh/geo`` directory and run the following command:

.. code-block:: bash

   gmsh -2 foc.geo -o foc.msh

Implementation
--------------------

First we load the appropriate libraries and initialize the mesh. We also define no-slip and zero pressure boundary conditions.
Note the relative path from the demo script to the generated ``foc.msh`` mesh file through the GMSH interface. 

.. code-block:: python 

    import dolfinx
    import numpy as np

    from flatiron_tk.mesh import Mesh
    from flatiron_tk.physics import TransientNavierStokes
    from flatiron_tk.solver  import ConvergenceMonitor
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

    # Define boundary condition functions
    def no_slip(x):
        return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

    def zero_pressure(x):
        return np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type)

    # Define the mesh
    mesh_file = '../mesh/foc.msh'
    mesh = Mesh(mesh_file=mesh_file)

Next we build the time dependent incompressible Navier-Stokes solver. Here we will use the unstable Q1P1 elements with SUPG 
and PSPG stabilization. We use a short-hand for stabilization where we set ``stab=True`` within the ``set_weak_form`` method directly.

.. code-block:: python 

    # Create transient Navier-Stokes object
    nse = TransientNavierStokes(mesh)
    nse.set_element('CG', 1, 'CG', 1)
    nse.build_function_space()

    # Physical parameters
    dt = 0.05
    mu = 0.001
    rho = 1
    u_mag = 4 

    nse.set_time_step_size(dt)
    nse.set_midpoint_theta(0.5)
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)
    nse.set_weak_form(stab=True)

Next, we define the boundary conditions. We use get the function spaces for the velocity and pressure boundary conditions from the 
mixed function space in the Navier-Stokes object. We then define the time-dependent inlet velocity expression and set the boundary conditions.

.. code-block:: python 

    # Get function spaces for boundary conditions functions
    V_u = nse.get_function_space('u').collapse()[0]
    V_p = nse.get_function_space('p').collapse()[0]

    # Parabolic profile 
    def inlet_velocity(x):
        # Parabolic profile: u_x = 4 * U_max * y * (H - y) / H^2
        # Assuming inlet along x, y in [0, H], U_max = 10.0, H = 4.1
        values = np.zeros((2, x.shape[1]), dtype=dolfinx.default_scalar_type)
        y = x[1]
        H = 4.1
        U_max = u_mag
        values[0] = 4 * U_max * y * (H - y) / (H ** 2)
        return values

    inlet_v = dolfinx.fem.Function(V_u)
    inlet_v.interpolate(lambda x: inlet_velocity(x))

    zero_p = dolfinx.fem.Function(V_p)
    zero_p.interpolate(zero_pressure)

    zero_v = dolfinx.fem.Function(V_u)
    zero_v.interpolate(no_slip)

    u_bcs = {
            1: {'type': 'dirichlet', 'value': inlet_v},
            2: {'type': 'dirichlet', 'value': zero_v},
            4: {'type': 'dirichlet', 'value': zero_v},
            5: {'type': 'dirichlet', 'value': zero_v}
            }

    p_bcs = {
            3: {'type': 'dirichlet', 'value': zero_p},
            }

    bc_dict = {'u': u_bcs, 
            'p': p_bcs}

    nse.set_bcs(bc_dict)


Next we set the output writer and define the NonLinear solver. We will use a Krylov solver with a LU preconditioner.

.. code-block:: python 

    # Set the output writer
    nse.set_writer('output', 'pvd')

    # Set the problem 
    problem = NonLinearProblem(nse)

    # Set the solver
    def my_custom_ksp_setup(ksp):
        ksp.setType(ksp.Type.FGMRES)        
        ksp.pc.setType(ksp.pc.Type.LU)  
        ksp.setTolerances(rtol=1e-12, atol=1e-10, max_it=500)
        ksp.setMonitor(ConvergenceMonitor('ksp'))

    solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)

Finally, we run the time-stepping loop. We will run the simulation until :math:`T=10` seconds.

.. code-block:: python 

    # Solve
    while t < 10.0:
        print(f'Solving at time t = {t:.2f}')
        
        # Set the inlet velocity for the current time step
        inlet_v.interpolate(lambda x: inlet_velocity(x))
        
        # Solve the problem
        solver.solve()

        nse.update_previous_solution()
        nse.write(time_stamp=t)
        
        # Update time
        t += dt

Full Script
--------------------

.. code-block:: python 

    import dolfinx
    import numpy as np

    from flatiron_tk.mesh import Mesh
    from flatiron_tk.physics import TransientNavierStokes
    from flatiron_tk.solver  import ConvergenceMonitor
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

    # Define boundary condition functions
    def no_slip(x):
        return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

    def zero_pressure(x):
        return np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type)

    # Define the mesh
    mesh_file = '../mesh/foc.msh'
    mesh = Mesh(mesh_file=mesh_file)

    # Create transient Navier-Stokes object
    nse = TransientNavierStokes(mesh)
    nse.set_element('CG', 1, 'CG', 1)
    nse.build_function_space()

    # Physical parameters
    dt = 0.05
    mu = 0.001
    rho = 1
    u_mag = 4 

    nse.set_time_step_size(dt)
    nse.set_midpoint_theta(0.5)
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)
    nse.set_weak_form(stab=True)

    # Get function spaces for boundary conditions functions
    V_u = nse.get_function_space('u').collapse()[0]
    V_p = nse.get_function_space('p').collapse()[0]

    # Parabolic profile 
    def inlet_velocity(x):
        # Parabolic profile: u_x = 4 * U_max * y * (H - y) / H^2
        # Assuming inlet along x, y in [0, H], U_max = 10.0, H = 4.1
        values = np.zeros((2, x.shape[1]), dtype=dolfinx.default_scalar_type)
        y = x[1]
        H = 4.1
        U_max = u_mag
        values[0] = 4 * U_max * y * (H - y) / (H ** 2)
        return values

    inlet_v = dolfinx.fem.Function(V_u)
    inlet_v.interpolate(lambda x: inlet_velocity(x))

    zero_p = dolfinx.fem.Function(V_p)
    zero_p.interpolate(zero_pressure)

    zero_v = dolfinx.fem.Function(V_u)
    zero_v.interpolate(no_slip)

    u_bcs = {
            1: {'type': 'dirichlet', 'value': inlet_v},
            2: {'type': 'dirichlet', 'value': zero_v},
            4: {'type': 'dirichlet', 'value': zero_v},
            5: {'type': 'dirichlet', 'value': zero_v}
            }

    p_bcs = {
            3: {'type': 'dirichlet', 'value': zero_p},
            }

    bc_dict = {'u': u_bcs, 
            'p': p_bcs}

    nse.set_bcs(bc_dict)

    # Set the output writer
    nse.set_writer('output', 'pvd')

    # Set the problem 
    problem = NonLinearProblem(nse)

    # Set the solver
    def my_custom_ksp_setup(ksp):
        ksp.setType(ksp.Type.FGMRES)        
        ksp.pc.setType(ksp.pc.Type.LU)  
        ksp.setTolerances(rtol=1e-12, atol=1e-10, max_it=500)
        ksp.setMonitor(ConvergenceMonitor('ksp'))

    solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)

    # Solve
    while t < 10.0:
        print(f'Solving at time t = {t:.2f}')
        
        # Set the inlet velocity for the current time step
        inlet_v.interpolate(lambda x: inlet_velocity(x))
        
        # Solve the problem
        solver.solve()

        nse.update_previous_solution()
        nse.write(time_stamp=t)
        
        # Update time
        t += dt
