============================================
Demo: Massless Tracers 
===========================================

This demo illustrates the usage of massless tracers in a fluid flow simulation using our library.
We will simulate the advection of massless tracer particles in a 2D fluid domain. 

Tracers on a steady velocity field
----------------------------------------
Tracers can be used in both steady and unsteady velocity fields. In `demo_steady_massless_tracers.py`, we
use the velocity field from a lid driven cavity flow simulation. The description of the fluid 
setup can be found in  :doc:`demo_steady_navier_stokes`. 

We being by importing the necessary modules:

.. code-block:: python

    import numpy as np
    import dolfinx
    from flatiron_tk.mesh import RectMesh
    from flatiron_tk.physics import SteadyNavierStokes
    from flatiron_tk.physics import MasslessTracerTracker

    from flatiron_tk.solver  import ConvergenceMonitor
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

We then define a function to create the mesh and solve the steady Navier-Stokes equations:

.. code-block:: python

    def solve_nse(mesh, Re):
        # Define boundary conditions functions
        def no_slip(x):
            return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

        def u_inlet(x):
            return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

        # Build Navier-Stokes problem
        nse = SteadyNavierStokes(mesh)
        nse.set_element('CG', 1, 'CG', 1)
        nse.build_function_space()

        # Set physical parameters
        mu = 1.0 / Re
        rho = 1.0
        nse.set_density(rho)
        nse.set_dynamic_viscosity(mu)

        # Set weak form and stabilization
        nse.set_weak_form()
        nse.add_stab()

        # Velocity and pressure subspaces
        V_u = nse.get_function_space('u').collapse()[0]
        V_p = nse.get_function_space('p').collapse()[0]

        # Create boundary condition functions
        zero_v = dolfinx.fem.Function(V_u)
        zero_v.interpolate(no_slip)
        inlet_v = dolfinx.fem.Function(V_u)
        inlet_v.interpolate(u_inlet)
        zero_p = dolfinx.fem.Function(V_p)
        zero_p.x.array[:] = 0.0

        # Define boundary conditions
        u_bcs = {
                1: {'type': 'dirichlet', 'value': zero_v},
                2: {'type': 'dirichlet', 'value': zero_v},
                3: {'type': 'dirichlet', 'value': zero_v},
                4: {'type': 'dirichlet', 'value': inlet_v},
                }

        p_bcs = {
                1: {'type': 'dirichlet', 'value': zero_p},
                }

        bc_dict = {'u': u_bcs, 
                'p': p_bcs}

        nse.set_bcs(bc_dict)

        nse.set_writer('output', 'pvd')

        # Define problem
        problem = NonLinearProblem(nse)

        # Custom KSP setup function
        def my_custom_ksp_setup(ksp):
            ksp.setType(ksp.Type.FGMRES)        
            ksp.pc.setType(ksp.pc.Type.LU)  
            ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)
            ksp.setMonitor(ConvergenceMonitor('ksp'))

        # Create nonlinear solver
        solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)
        
        # Solve the problem
        solver.solve()
        nse.write()

        return nse

    # Create mesh
    ne = 64
    mesh = RectMesh(0.0, 0.0, 1.0, 1.0, 1/ne)

    nse = solve_nse(mesh=mesh, Re=100.0)

Since we are only using the velocity field from a steady simulation, we can simply return the Navier-Stokes object for use in 
the tracer simulation. 

We not set up the tracer tracker and writer. Particles are saved in VTK format for visualization in Paraview.

.. code-block:: python

    particle_tracker = MasslessTracerTracker(mesh, dt=0.01)
    particle_tracker.set_writer('output/particles')

We have two options for seeding particle. We can either seed particles from an array using the `set_particle_positions`
method, or seed particles from a marked boundary using the `set_particle_positions_from_boundary` method. This method seeds 
a particle on each node of the specified boundary. In this example, we will seed particles from the top boundary (boundary ID 4), the inlet of the 
lid driven cavity. We also move the particles slightly into the domainsuch that they will experience some movement in the :math:`\hat{y}` direction. 
We will then write the initial particle positions to file.

.. code-block:: python

    # Seed particles from the inlet boundary (boundary ID 4)
    particle_tracker.set_particle_positions_from_boundary(boundary_id=4)
    particle_tracker.particle_positions[:, 1] -= 0.1 

    # Write initial particle positions
    particle_tracker.write()


Finally, we extract the velocity field from the Navier-Stokes solution and update the particle positions in a loop. We write the particle positions
to file at each time step. We use the simple Euler method for time integration.

.. code-block:: python

    # Extract velocity field and update particle positions
    u = nse.get_solution_function().sub(0)
    for i in range(1000):
        particle_tracker.update_particle_positions(current_velocity=u, method='euler')
        particle_tracker.write(time_stamp=i)


Tracers on an unsteady velocity field
----------------------------------------
In `demo_transient_massless_tracers.py`, we use a time-dependent velocity field. The Navier-Stokes demo is described in :doc:`demo_transient_navier_stokes`.
We begin by importing the necessary modules:

.. code-block:: python

    import dolfinx
    import flatiron_tk
    import numpy as np

    from flatiron_tk.mesh import Mesh
    from flatiron_tk.physics import MasslessTracerTracker
    from flatiron_tk.physics import TransientNavierStokes
    from flatiron_tk.solver  import ConvergenceMonitor
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

Next, we define the flow over cylinder problem as described in :doc:`demo_transient_navier_stokes`. 

.. code-block:: python 

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

    # Set physics object parameters
    nse.set_time_step_size(dt)
    nse.set_midpoint_theta(0.5)
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)
    nse.set_weak_form(stab=True)

    # Get function spaces for boundary conditions functions
    V_u = nse.get_function_space('u').collapse()[0]
    V_p = nse.get_function_space('p').collapse()[0]

    # Set boundary conditions
    profile = flatiron_tk.ParabolicInletProfile(flow_rate=2.0/3.0, radius=4.1/2.0, center=mesh.get_boundary_centroid(1), normal=np.array([1.0, 0.0]))
    inlet_v = dolfinx.fem.Function(V_u)
    inlet_v.interpolate(profile)

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

We set up the tracer tracker and writer as before, and write the initial particle positions. In this example, we seed particles from the left boundary (boundary ID 1).:

.. code-block:: python

    # Set up particle tracker
    particle_tracker = MasslessTracerTracker(mesh, dt=dt)
    particle_tracker.set_writer('output/particles')
    particle_tracker.set_particle_positions_from_boundary(boundary_id=1)

    # Write initial particle positions
    particle_tracker.write()

Finally, we run the time-stepping loop, updating the Navier-Stokes solution and the particle positions at each time step. Here, we use the 'heun' method for time integration of the particle positions.
For this method, we need both the current and previous velocity fields.


.. code-block:: python
    
    # Solve
    t = 0.0
    count = 0
    while t < 5:
        print(f'Solving at time t = {t:.2f}')
        
        # Solve the problem
        solver.solve()

        # Update and write solution
        nse.update_previous_solution()
        nse.write(time_stamp=t)

        # Get current and previous velocity fields
        u0 = nse.previous_solution.sub(0)
        un = nse.get_solution_function().sub(0)

        # Update particle positions
        particle_tracker.update_particle_positions(current_velocity=un, previous_velocity=u0, method='heun')
        particle_tracker.write(time_stamp=t)

Here, we inject particles continuously from the inlet boundary at every 5 time steps. Similar to the initial seed, we can 
inject new particles on each node of the specified boundary or by a given array of positions.

.. code-block:: python

        # Inject new particles every 5 time steps
        if count % 5 == 0:
            particle_tracker.set_particle_positions_from_boundary(boundary_id=1)

        # Update time and counter
        t += dt
        count += 1

Full Steady Script
----------------------------------------

.. code-block:: python 

    import numpy as np
    import dolfinx
    from flatiron_tk.mesh import RectMesh
    from flatiron_tk.physics import SteadyNavierStokes
    from flatiron_tk.physics import MasslessTracerTracker

    from flatiron_tk.solver  import ConvergenceMonitor
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

    def solve_nse(mesh, Re):
        # Define boundary conditions functions
        def no_slip(x):
            return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

        def u_inlet(x):
            return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

        # Build Navier-Stokes problem
        nse = SteadyNavierStokes(mesh)
        nse.set_element('CG', 1, 'CG', 1)
        nse.build_function_space()

        # Set physical parameters
        mu = 1.0 / Re
        rho = 1.0
        nse.set_density(rho)
        nse.set_dynamic_viscosity(mu)

        # Set weak form and stabilization
        nse.set_weak_form()
        nse.add_stab()

        # Velocity and pressure subspaces
        V_u = nse.get_function_space('u').collapse()[0]
        V_p = nse.get_function_space('p').collapse()[0]

        # Create boundary condition functions
        zero_v = dolfinx.fem.Function(V_u)
        zero_v.interpolate(no_slip)
        inlet_v = dolfinx.fem.Function(V_u)
        inlet_v.interpolate(u_inlet)
        zero_p = dolfinx.fem.Function(V_p)
        zero_p.x.array[:] = 0.0

        # Define boundary conditions
        u_bcs = {
                1: {'type': 'dirichlet', 'value': zero_v},
                2: {'type': 'dirichlet', 'value': zero_v},
                3: {'type': 'dirichlet', 'value': zero_v},
                4: {'type': 'dirichlet', 'value': inlet_v},
                }

        p_bcs = {
                1: {'type': 'dirichlet', 'value': zero_p},
                }

        bc_dict = {'u': u_bcs, 
                'p': p_bcs}

        nse.set_bcs(bc_dict)

        nse.set_writer('output', 'pvd')

        # Define problem
        problem = NonLinearProblem(nse)

        # Custom KSP setup function
        def my_custom_ksp_setup(ksp):
            ksp.setType(ksp.Type.FGMRES)        
            ksp.pc.setType(ksp.pc.Type.LU)  
            ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)
            ksp.setMonitor(ConvergenceMonitor('ksp'))

        # Create nonlinear solver
        solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)
        
        # Solve the problem
        solver.solve()
        nse.write()

        return nse

    # Create mesh
    ne = 64
    mesh = RectMesh(0.0, 0.0, 1.0, 1.0, 1/ne)

    nse = solve_nse(mesh=mesh, Re=100.0)

    # Set up particle tracker
    particle_tracker = MasslessTracerTracker(mesh, dt=0.01)
    particle_tracker.set_writer('output/particles')
    particle_tracker.set_particle_positions_from_boundary(boundary_id=4)

    # Shifting particles down to get them in the domain better (for lid driven cavity mesh)
    particle_tracker.particle_positions[:, 1] -= 0.1 

    # Write initial particle positions
    particle_tracker.write()



    u = nse.get_solution_function().sub(0)
    for i in range(1000):
        particle_tracker.update_particle_positions(current_velocity=u, method='euler')
        particle_tracker.write(time_stamp=i)

Full Unsteady Script
----------------------------------------

.. code-block:: python 

    import numpy as np
    import dolfinx
    from flatiron_tk.mesh import RectMesh
    from flatiron_tk.physics import SteadyNavierStokes
    from flatiron_tk.physics import MasslessTracerTracker

    from flatiron_tk.solver  import ConvergenceMonitor
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

    def solve_nse(mesh, Re):
        # Define boundary conditions functions
        def no_slip(x):
            return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

        def u_inlet(x):
            return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

        # Build Navier-Stokes problem
        nse = SteadyNavierStokes(mesh)
        nse.set_element('CG', 1, 'CG', 1)
        nse.build_function_space()

        # Set physical parameters
        mu = 1.0 / Re
        rho = 1.0
        nse.set_density(rho)
        nse.set_dynamic_viscosity(mu)

        # Set weak form and stabilization
        nse.set_weak_form()
        nse.add_stab()

        # Velocity and pressure subspaces
        V_u = nse.get_function_space('u').collapse()[0]
        V_p = nse.get_function_space('p').collapse()[0]

        # Create boundary condition functions
        zero_v = dolfinx.fem.Function(V_u)
        zero_v.interpolate(no_slip)
        inlet_v = dolfinx.fem.Function(V_u)
        inlet_v.interpolate(u_inlet)
        zero_p = dolfinx.fem.Function(V_p)
        zero_p.x.array[:] = 0.0

        # Define boundary conditions
        u_bcs = {
                1: {'type': 'dirichlet', 'value': zero_v},
                2: {'type': 'dirichlet', 'value': zero_v},
                3: {'type': 'dirichlet', 'value': zero_v},
                4: {'type': 'dirichlet', 'value': inlet_v},
                }

        p_bcs = {
                1: {'type': 'dirichlet', 'value': zero_p},
                }

        bc_dict = {'u': u_bcs, 
                'p': p_bcs}

        nse.set_bcs(bc_dict)

        nse.set_writer('output', 'pvd')

        # Define problem
        problem = NonLinearProblem(nse)

        # Custom KSP setup function
        def my_custom_ksp_setup(ksp):
            ksp.setType(ksp.Type.FGMRES)        
            ksp.pc.setType(ksp.pc.Type.LU)  
            ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)
            ksp.setMonitor(ConvergenceMonitor('ksp'))

        # Create nonlinear solver
        solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)
        
        # Solve the problem
        solver.solve()
        nse.write()

        return nse

    # Create mesh
    ne = 64
    mesh = RectMesh(0.0, 0.0, 1.0, 1.0, 1/ne)

    nse = solve_nse(mesh=mesh, Re=100.0)

    # Set up particle tracker
    particle_tracker = MasslessTracerTracker(mesh, dt=0.01)
    particle_tracker.set_writer('output/particles')
    particle_tracker.set_particle_positions_from_boundary(boundary_id=4)

    # Shifting particles down to get them in the domain better (for lid driven cavity mesh)
    particle_tracker.particle_positions[:, 1] -= 0.1 

    # Write initial particle positions
    particle_tracker.write()



    u = nse.get_solution_function().sub(0)
    for i in range(1000):
        particle_tracker.update_particle_positions(current_velocity=u, method='euler')
        particle_tracker.write(time_stamp=i)