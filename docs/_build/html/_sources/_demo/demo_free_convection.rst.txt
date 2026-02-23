==============================================
Demo: Free Convection (Thermofluid Coupling)
=============================================

In this demo, we demonstrate how to solve a coupled thermofluid problem using the Boussinesq approximation. The source code can be found in ``demo/demo_free_convection/demo_free_convection.py``.
This problem solves the 2D free convection problem in a square cavity. The strong form of the problem is given by the incompressible Navier-Stokes equation coupled with the heat equation:

.. math::

    \rho \frac{\partial \textbf{u}}{\partial t} + \rho \textbf{u} \cdot \nabla \textbf{u} = \nabla \cdot \boldsymbol{\sigma} + \textbf{b}

and

.. math::

    \nabla \cdot \textbf{u} = 0

and

.. math::

    \frac{\partial T}{\partial t} + \textbf{u} \cdot \nabla T = \alpha \nabla^2 T + Q

where :math:`\textbf{u}` is the velocity field, :math:`\boldsymbol{\sigma}` is the Cauchy stress tensor, :math:`T` is the temperature field, :math:`\rho` is 
the density, :math:`\alpha` is the thermal diffusivity, and :math:`Q` is a heat source term. 
The body force :math:`\textbf{b}` includes the buoyancy term due to temperature variations and is given by:

.. math::

    \textbf{b} = \rho \textbf{g} \beta (T - T_{ref})

where :math:`\textbf{g}` is the gravitational acceleration, :math:`\beta` is the thermal expansion coefficient, and :math:`T_{ref}` is a reference temperature.

The problem is defined in a square domain :math:`(x,y) \in [0,1] \times [0,1]`. The boundary conditions are as follows:
- Left wall (x=0): No-slip for velocity (:math:`\textbf{u} = \textbf{0}`) and hot temperature (:math:`T = T_{hot}`).
- Right wall (x=1): No-slip for velocity (:math:`\textbf{u} = \textbf{0}`) and cold temperature (:math:`T = T_{cold}`).
- Top wall (y=1) and bottom wall (y=0): No-slip for velocity (:math:`\textbf{u} = \textbf{0}`) and adiabatic for temperature (:math:`\frac{\partial T}{\partial n} = 0`).

We begin by importing the necessary libraries and defining the problem parameters, including the physical properties of the fluid, the gravitational acceleration, and the temperature boundary conditions.

.. code-block::

    import dolfinx
    import flatiron_tk
    import numpy as np
    import ufl

    from flatiron_tk.mesh import RectMesh
    from flatiron_tk.physics import TransientNavierStokes
    from flatiron_tk.physics import TransientScalarTransport
    from flatiron_tk.physics import TransientMultiPhysicsProblem
    from flatiron_tk.solver import ConvergenceMonitor
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

    # Define fluid properties
    dt = 0.001
    rho = 1.0
    mu = 1e-2 

    # Define thermal properties
    Pr = 1
    Ra = 1e5 
    delta_temp = 1.0
    length = 1.0
    gravity = -9.81
    specific_heat = 1.0

    alpha = mu / (rho * Pr)
    conductivity = alpha * rho * specific_heat
    expansion_coef = (mu * conductivity * Ra) / (rho * gravity * delta_temp * length**3)

We then create a unit square mesh using the built-in meshing capabilities of Flatiron.

.. code-block:: python

    # Define the mesh
    ne = 64
    h = 1/ne
    mesh = RectMesh(0, 0, 1, 1, h)

Next, we create the Navier-Stokes and heat transfer physics objects, specifying the relevant parameters. Additionally, we set up the multiphysics problem to strongly couple the two physics.

.. code-block:: python 

    # Create transient Navier-Stokes and scalar transport physics
    nse = TransientNavierStokes(mesh)
    nse.set_element('CG', 1, 'CG', 1)
    nse.build_function_space()
    nse.set_time_step_size(dt)
    nse.set_midpoint_theta(0.5)
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)

    adr = TransientScalarTransport(mesh)
    adr.set_tag('T')
    adr.set_element('CG', 1)
    adr.set_time_step_size(dt)
    adr.set_diffusivity(alpha, alpha)
    adr.set_reaction(0.0, 0.0)

    coupled_physics = TransientMultiPhysicsProblem(nse, adr)
    coupled_physics.set_element()
    coupled_physics.build_function_space()

We extract the necessary test and trial functions for coupling the momentum and energy equations. 

.. code-block:: python 

    # Get functions and test/trial functions for later
    p = nse.get_solution_function('p')
    q = nse.get_test_function('p')
    u = nse.get_solution_function('u')
    u0 = ufl.split(coupled_physics.sub_physics[0].previous_solution)[0]
    T = adr.get_solution_function('T')
    T0 = ufl.split(coupled_physics.sub_physics[1].previous_solution)[0]
    w = coupled_physics.sub_physics[0].get_test_function('u')  

We then using the solution functions from the Navier-Stokes equations to coupled the scalar transport advection velocity. Once this has been set, we 
set the basic weak forms for both physics.

.. code-block:: python 

    # Set weak forms
    adr.set_advection_velocity(u0, u0)
    nse_options = {'stab': True}
    adr_options = {'stab': True}
    coupled_physics.set_weak_form(nse_options, adr_options)

The Boussinesq approximation is implemented by adding a buoyancy term to the momentum equation. Since this form is not 
standard in the Navier-Stokes class, we manually add this term to the weak form of the coupled physics.

.. code-block:: python 

    # Boussinesq force
    g = ufl.as_vector([0.0, gravity])
    boussinesq_term = 0.5 * ((1 - T) * expansion_coef * g + (1 - T0) * expansion_coef * g)

    # Add to weak form
    coupled_physics.add_to_weak_form(ufl.inner(boussinesq_term, w) * nse.dx)

Next, we define the boundary conditions for both physics. The velocity field has no-slip conditions on all walls, while the temperature field has hot and cold Dirichlet conditions on the left and right walls, respectively, and adiabatic (Neumann) conditions on the top and bottom walls.
We bound pressure through an additional penalty condition. 

.. code-block:: python 

    V_u = nse.get_function_space('u').collapse()[0]
    V_p = nse.get_function_space('p').collapse()[0]
    V_T = adr.get_function_space().collapse()[0]

    def no_slip_bc(x):
        """No-slip boundary condition for velocity (u=0)."""
        return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

    def lid_bc(x):
        """Lid-driven cavity velocity profile."""
        vals = np.zeros((mesh.msh.geometry.dim, x.shape[1]))
        vals[0] = 1.0
        vals[1] = 0.0
        return vals

    lid_v = dolfinx.fem.Function(V_u)
    lid_v.interpolate(lid_bc)

    zero_v = dolfinx.fem.Function(V_u)
    zero_v.interpolate(no_slip_bc)

    zero_p = flatiron_tk.constant(mesh, 0.0)

    u_bcs = {1: {'type': 'dirichlet', 'value': zero_v},
            2: {'type': 'dirichlet', 'value': zero_v},
            3: {'type': 'dirichlet', 'value': zero_v},
            4: {'type': 'dirichlet', 'value': zero_v}}

    p_bcs = {}

    p_ref = flatiron_tk.constant(mesh, 0.0)
    eps = 1e-10
    pressure_penalty = eps * ufl.inner(p - p_ref, q) * nse.dx
    coupled_physics.add_to_weak_form(pressure_penalty)

    T_bcs = {1: {'type': 'dirichlet', 'value': flatiron_tk.constant(mesh, 0.5*delta_temp)},
            3: {'type': 'dirichlet', 'value': flatiron_tk.constant(mesh, -0.5*delta_temp)},}

    bc_dict = {
        'u': u_bcs,
        'p': p_bcs,
        'T': T_bcs
    }
    coupled_physics.set_bcs(bc_dict)

Finally, we set up problem, solver, and writer. Then, the time-stepping loop is executed to solve the coupled thermofluid problem. 

.. code-block:: python

    problem = NonLinearProblem(coupled_physics)
    coupled_physics.set_writer('output', 'xdmf')

    problem = NonLinearProblem(coupled_physics)
    def my_custom_ksp_setup(ksp):
        ksp.setType(ksp.Type.FGMRES)        
        ksp.pc.setType(ksp.pc.Type.LU)  
        ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)
        ksp.setMonitor(ConvergenceMonitor('ksp'))

    solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)

    num_steps = 1000
    for step in range(num_steps):
        print(f'Solving time step {step+1}/{num_steps}, Time: {(step+1)*dt:.4f}')
        solver.solve()

        coupled_physics.write()
        coupled_physics.update_previous_solution()

Full script:
--------------

.. code-block:: python 

    import dolfinx
    import flatiron_tk
    import numpy as np
    import ufl

    from flatiron_tk.mesh import RectMesh
    from flatiron_tk.physics import TransientNavierStokes
    from flatiron_tk.physics import TransientScalarTransport
    from flatiron_tk.physics import TransientMultiPhysicsProblem
    from flatiron_tk.solver import ConvergenceMonitor
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

    # Define fluid properties
    dt = 0.001
    rho = 1.0
    mu = 1e-2 

    # Define thermal properties
    Pr = 1
    Ra = 1e5 
    delta_temp = 1.0
    length = 1.0
    gravity = -9.81
    specific_heat = 1.0

    alpha = mu / (rho * Pr)
    conductivity = alpha * rho * specific_heat
    expansion_coef = (mu * conductivity * Ra) / (rho * gravity * delta_temp * length**3)

    # Define the mesh
    ne = 64
    h = 1/ne
    mesh = RectMesh(0, 0, 1, 1, h)

    # Create transient Navier-Stokes and scalar transport physics
    nse = TransientNavierStokes(mesh)
    nse.set_element('CG', 1, 'CG', 1)
    nse.build_function_space()
    nse.set_time_step_size(dt)
    nse.set_midpoint_theta(0.5)
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)

    adr = TransientScalarTransport(mesh)
    adr.set_tag('T')
    adr.set_element('CG', 1)
    adr.set_time_step_size(dt)
    adr.set_diffusivity(alpha, alpha)
    adr.set_reaction(0.0, 0.0)

    coupled_physics = TransientMultiPhysicsProblem(nse, adr)
    coupled_physics.set_element()
    coupled_physics.build_function_space()

    # Get functions and test/trial functions for later
    p = nse.get_solution_function('p')
    q = nse.get_test_function('p')
    u = nse.get_solution_function('u')
    u0 = ufl.split(coupled_physics.sub_physics[0].previous_solution)[0]
    T = adr.get_solution_function('T')
    T0 = ufl.split(coupled_physics.sub_physics[1].previous_solution)[0]
    w = coupled_physics.sub_physics[0].get_test_function('u') 

    # Set weak forms
    adr.set_advection_velocity(u0, u0)
    nse_options = {'stab': True}
    adr_options = {'stab': True}
    coupled_physics.set_weak_form(nse_options, adr_options)

    # Boussinesq force
    g = ufl.as_vector([0.0, gravity])
    boussinesq_term = 0.5 * ((1 - T) * expansion_coef * g + (1 - T0) * expansion_coef * g)

    # Add to weak form
    coupled_physics.add_to_weak_form(ufl.inner(boussinesq_term, w) * nse.dx)

    V_u = nse.get_function_space('u').collapse()[0]
    V_p = nse.get_function_space('p').collapse()[0]
    V_T = adr.get_function_space().collapse()[0]

    def no_slip_bc(x):
        """No-slip boundary condition for velocity (u=0)."""
        return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

    def lid_bc(x):
        """Lid-driven cavity velocity profile."""
        vals = np.zeros((mesh.msh.geometry.dim, x.shape[1]))
        vals[0] = 1.0
        vals[1] = 0.0
        return vals

    lid_v = dolfinx.fem.Function(V_u)
    lid_v.interpolate(lid_bc)

    zero_v = dolfinx.fem.Function(V_u)
    zero_v.interpolate(no_slip_bc)

    zero_p = flatiron_tk.constant(mesh, 0.0)

    u_bcs = {1: {'type': 'dirichlet', 'value': zero_v},
            2: {'type': 'dirichlet', 'value': zero_v},
            3: {'type': 'dirichlet', 'value': zero_v},
            4: {'type': 'dirichlet', 'value': zero_v}}

    p_bcs = {}

    p_ref = flatiron_tk.constant(mesh, 0.0)
    eps = 1e-10
    pressure_penalty = eps * ufl.inner(p - p_ref, q) * nse.dx
    coupled_physics.add_to_weak_form(pressure_penalty)

    T_bcs = {1: {'type': 'dirichlet', 'value': flatiron_tk.constant(mesh, 0.5*delta_temp)},
            3: {'type': 'dirichlet', 'value': flatiron_tk.constant(mesh, -0.5*delta_temp)},}

    bc_dict = {
        'u': u_bcs,
        'p': p_bcs,
        'T': T_bcs
    }

    coupled_physics.set_bcs(bc_dict)

    problem = NonLinearProblem(coupled_physics)
    coupled_physics.set_writer('output', 'xdmf')

    problem = NonLinearProblem(coupled_physics)
    def my_custom_ksp_setup(ksp):
        ksp.setType(ksp.Type.FGMRES)        
        ksp.pc.setType(ksp.pc.Type.LU)  
        ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)
        ksp.setMonitor(ConvergenceMonitor('ksp'))

    solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)

    num_steps = 1000
    for step in range(num_steps):
        print(f'Solving time step {step+1}/{num_steps}, Time: {(step+1)*dt:.4f}')
        solver.solve()

        coupled_physics.write()
        coupled_physics.update_previous_solution()