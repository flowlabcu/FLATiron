==============================================================
Demo: Transient Advection-Diffusion-Reaction Problem
==============================================================

This demo demonstrates how to solve a 1D transient Advection-Diffusion-Reaction problem. 
The full source code can be found in **demo/demo_transient_adr/demo_transient_adr_1D.py**.

The following problem was taken from Problem 1 in this `paper <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/CE047p0241>`_

Problem definition
--------------------

The strong form:

.. math::

	\frac{\partial c}{\partial t} + u \frac{\partial c}{\partial c} = D \frac{\partial^2 c}{\partial x^2}

In this demo, we solve a transport of a Gaussian pulse under a time-varying advection velocity :math:`u(t)` and a fixed diffusivity :math:`D`. The analytical solution of this problem is

.. math::

	c(x,t) = \frac{\sigma_0}{\sigma}exp\left(-\frac{(x-\bar{x})^2}{2\sigma^2}\right) \\

where

.. math::

	\sigma^2 = \sigma_0^2 +2Dt \\

	\bar{x} = x_0 + \int_0^t u(\tau) d\tau \\

and :math:`\sigma_0` and :math:`x_0` are constant initial Gaussian standard deviation and center respectively. 
From the analytical solution, one can devise the initial condition by simply substituting :math:`t=0` in the above expressions.

In the `paper <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/CE047p0241>`_, the boundary 
condition is :math:`c(x,t) \rightarrow 0` as :math:`|x| \rightarrow \infty`. To approximate this boundary 
condition in a finite domain, we will set :math:`c(x,t)=0` at both boundary points, and ensure that the 
Gaussian pulse is sufficiently far enough from both boundaries.

Implementation
--------------------

We first import relevant libraries and define a mesh using the LineMesh object.

.. code-block:: python

    import dolfinx
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    import ufl

    from flatiron_tk.mesh import LineMesh
    from flatiron_tk.physics import TransientScalarTransport
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver
    
    # Create a Line Mesh 
    num_elements = 128 
    h = 12800 / num_elements
    mesh = LineMesh(0, 12800, h)

Next, we define several functions for the exact solution of the problem. Note 
that we will use the exact solution to create the initial condition for our numerical setup.

.. code-block:: python

    # Functions for the exact solution
    def get_x_bar(a_a, a_b, a_t):
        x_bar = (a_a - a_a * np.cos(a_b * a_t)) / a_b
        return x_bar

    def get_sigma(a_sigma_0, a_D, a_t):
        sigma = np.sqrt(a_sigma_0**2 + 2 * a_D * a_t)
        return sigma

    def get_c_exact(a_x, a_a, a_b, a_t, a_D, a_sigma_0):
        sigma = get_sigma(a_sigma_0, a_D, a_t)
        x_bar = get_x_bar(a_a, a_b, a_t)
        c = (a_sigma_0 / sigma) * np.exp(-1 * (a_x - x_bar)**2 / (2 * sigma**2))
        return c

Next, we define the relevant physics. In this case, it is the ``TransientScalarTransport`` physics. We will integrate this problem in 
time using :math:`\Delta t = 96` similar to the problem in the benchmark `paper <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/CE047p0241>`_. 
It is noted that this method integrates the problem using the Crank-Nicolson midpoint algorithm. 
Please see :doc:`Transient Scalar Transport Physics <../_modules/physics_modules/module_scalar_transport>` for the time discretization details.

.. code-block:: python 

    # Define problem
    dt = 96
    a = 1.5
    b = 2 * np.pi / 9600
    time = dolfinx.fem.Constant(mesh.msh, 0.0)

    stp = TransientScalarTransport(mesh, dt, tag='c')
    stp.set_element('CG', 1)
    stp.build_function_space()
    V = stp.get_function_space()

Next, we define the advection, diffusion, and reaction terms. We use `dlfinx.fem.Expression` to interpolate the time-varying advection velocity.

.. code-block:: python 

    # Set diffusivity 
    diffusivity = 2.0
    stp.set_diffusivity(diffusivity, diffusivity)

    # Create a function for the advection velocity
    u0 = dolfinx.fem.Function(V)
    un = dolfinx.fem.Function(V)
    u0.name = 'u0'
    un.name = 'un'

    # Interpoate a ufl expression for the advection velocity
    u_expr = a * ufl.sin(b * time)
    interpolation_points = V.element.interpolation_points()
    u0.interpolate(dolfinx.fem.Expression(u_expr, interpolation_points))
    un.interpolate(dolfinx.fem.Expression(u_expr, interpolation_points))
    stp.set_advection_velocity(u0, un)

    # Set reaction term
    stp.set_reaction(0.0, 0.0)

We then set the weak form of the problem, and define the initial and boundary conditions.

.. code-block:: python

    # Set weak form and stabilization
    stp.set_weak_form()
    stp.add_stab()

    # Set intial condition
    x = ufl.SpatialCoordinate(mesh.msh)
    t_0 = 1000
    sigma_0 = 264
    sigma = get_sigma(sigma_0, diffusivity, t_0)
    x_bar = get_x_bar(a, b, t_0)
    c0 = dolfinx.fem.Function(V)
    c0_expr = (sigma_0 / sigma) * ufl.exp(-(x[0] - x_bar)**2 / (2 * sigma**2))
    c0.interpolate(dolfinx.fem.Expression(c0_expr, interpolation_points))
    stp.set_initial_condition(c0)

    # Set boundary conditions
    bc_dict = {1: {'type': 'dirichlet', 'value': dolfinx.fem.Constant(mesh.msh, 0.0)},
            2: {'type': 'dirichlet', 'value': dolfinx.fem.Constant(mesh.msh, 0.0)}}
    stp.set_bcs(bc_dict)

Next, we create a nonlinear solver to solve the problem at each time step.

.. code-block:: python

    # Set up the solver
    problem = NonLinearProblem(stp)
    solver = NonLinearSolver(mesh.msh.comm, problem)
    stp.set_writer('output', 'xdmf')

At each time step, the time-marching problem is solved by first updating the time-varying advection velocity. 
The scalar transport problem is then solved, and the solution is updated using the ``update_previous_solution()`` method within 
the ``TransientScalarTransport`` class. 
Finally, the solution for the current time step is written using ``stp.write(time_stamp=t)`` and visualized with `matplotlib`.

.. code-block:: python 

    # Set up the time-stepping
    t = t_0
    u_vals = []
    t_vals = []

    # Create an array for the x values to plot the solutions against
    x_plt = np.linspace(0, 12800, num_elements + 1, endpoint=True)
    # Create lists to hold the numerical and exact solutions at each time step
    c_vals_list = []
    sol_exact_list = []
    time_vals = []

    while t < 8000:
        # Update advection velocity
        time.value = t # Update ufl time expression
        u0.interpolate(dolfinx.fem.Expression(a * ufl.sin(b * time), interpolation_points))
        time.value = t + dt
        un.interpolate(dolfinx.fem.Expression(a * ufl.sin(b * time), interpolation_points))

        # Solve the current time step
        solver.solve()
        stp.update_previous_solution()

        # Write the solution 
        stp.write(time_stamp=t)

        # Plot the numerical and exact solutions
        sol_exact = get_c_exact(x_plt, a, b, t, diffusivity, sigma_0)
        c_vals = stp.get_solution_function().x.array

        c_vals = stp.get_solution_function().x.array.copy()
        sol_exact = get_c_exact(x_plt, a, b, t, diffusivity, sigma_0)

        c_vals_list.append(c_vals)
        sol_exact_list.append(sol_exact)
        time_vals.append(t)

        # Step forward in time
        t += dt

Plotting:

.. code-block:: python

    # Plot the evolution of the solution over time using matplotlib animation
    fig, ax = plt.subplots()
    line_num, = ax.plot([], [], label='Numerical')
    line_exact, = ax.plot([], [], label='Exact')
    ax.set_xlim(0, 12800)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

    def update(frame):
        line_num.set_data(x_plt, c_vals_list[frame])
        line_exact.set_data(x_plt, sol_exact_list[frame])
        ax.set_title(f"t = {time_vals[frame]:.0f}")
        return line_num, line_exact

    ani = animation.FuncAnimation(fig, update, frames=len(time_vals), blit=True, interval=100)
    ani.save('solution_evolution.mp4', writer='ffmpeg')
    plt.close(fig)

Full Script
--------------------
.. code-block:: python
    
    '''
    Demo for 1D transient convection-diffusion equation on an interval [0,12800]
    with no reactions
    dc/dt = D*d^2c/dx^2 - u*dc/dx - f

    The following problem was taken from "Problem 1" from "Benchmarks for the Transport Equation:
    The Convection_Diffusion Forum and Beyond" by Baptista and Adams, 1995

    D = 2
    f = 0
    u = 1.5*sin(2*pi*t/9600)
    The Gaussian Source Solution is:
    c(x,t) = sigma_0/sigma * exp(-(x-x_bar)^2 / 2*sigma^2)
    sigma^2 = sigma_0^2 + 2*D*t
    x_bar = x_0 + int(u(T)dT) from 0 to T

    This demo demonstrates how to do a transient convection-diffusion problem in flatiron_tk

    Author: njrovito
    '''
    import dolfinx
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    import ufl

    from flatiron_tk.mesh import LineMesh
    from flatiron_tk.physics import TransientScalarTransport
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

    # Functions for the exact solution
    def get_x_bar(a_a, a_b, a_t):
        x_bar = (a_a - a_a * np.cos(a_b * a_t)) / a_b
        return x_bar

    def get_sigma(a_sigma_0, a_D, a_t):
        sigma = np.sqrt(a_sigma_0**2 + 2 * a_D * a_t)
        return sigma

    def get_c_exact(a_x, a_a, a_b, a_t, a_D, a_sigma_0):
        sigma = get_sigma(a_sigma_0, a_D, a_t)
        x_bar = get_x_bar(a_a, a_b, a_t)
        c = (a_sigma_0 / sigma) * np.exp(-1 * (a_x - x_bar)**2 / (2 * sigma**2))
        return c

    # Create a Line Mesh 
    num_elements = 128 
    h = 12800 / num_elements
    mesh = LineMesh(0, 12800, h)

    # Define problem
    dt = 96
    a = 1.5
    b = 2 * np.pi / 9600
    time = dolfinx.fem.Constant(mesh.msh, 0.0)

    stp = TransientScalarTransport(mesh, dt, tag='c')
    stp.set_element('CG', 1)
    stp.build_function_space()
    V = stp.get_function_space()

    # Set diffusivity 
    diffusivity = 2.0
    stp.set_diffusivity(diffusivity, diffusivity)

    # Create a function for the advection velocity
    u0 = dolfinx.fem.Function(V)
    un = dolfinx.fem.Function(V)
    u0.name = 'u0'
    un.name = 'un'

    # Interpoate a ufl expression for the advection velocity
    u_expr = a * ufl.sin(b * time)
    interpolation_points = V.element.interpolation_points()
    u0.interpolate(dolfinx.fem.Expression(u_expr, interpolation_points))
    un.interpolate(dolfinx.fem.Expression(u_expr, interpolation_points))
    stp.set_advection_velocity(u0, un)

    # Set reaction term
    stp.set_reaction(0.0, 0.0)

    # Set weak form and stabilization
    stp.set_weak_form()
    stp.add_stab()

    # Set intial condition
    x = ufl.SpatialCoordinate(mesh.msh)
    t_0 = 1000
    sigma_0 = 264
    sigma = get_sigma(sigma_0, diffusivity, t_0)
    x_bar = get_x_bar(a, b, t_0)
    c0 = dolfinx.fem.Function(V)
    c0_expr = (sigma_0 / sigma) * ufl.exp(-(x[0] - x_bar)**2 / (2 * sigma**2))
    c0.interpolate(dolfinx.fem.Expression(c0_expr, interpolation_points))
    stp.set_initial_condition(c0)

    # Set boundary conditions
    bc_dict = {1: {'type': 'dirichlet', 'value': dolfinx.fem.Constant(mesh.msh, 0.0)},
            2: {'type': 'dirichlet', 'value': dolfinx.fem.Constant(mesh.msh, 0.0)}}
    stp.set_bcs(bc_dict)

    # Set up the solver
    problem = NonLinearProblem(stp)
    solver = NonLinearSolver(mesh.msh.comm, problem)
    stp.set_writer('output', 'xdmf')

    # Set up the time-stepping
    t = t_0
    u_vals = []
    t_vals = []

    # Create an array for the x values to plot the solutions against
    x_plt = np.linspace(0, 12800, num_elements + 1, endpoint=True)
    # Create lists to hold the numerical and exact solutions at each time step
    c_vals_list = []
    sol_exact_list = []
    time_vals = []

    while t < 8000:
        # Update advection velocity
        time.value = t # Update ufl time expression
        u0.interpolate(dolfinx.fem.Expression(a * ufl.sin(b * time), interpolation_points))
        time.value = t + dt
        un.interpolate(dolfinx.fem.Expression(a * ufl.sin(b * time), interpolation_points))

        # Solve the current time step
        solver.solve()
        stp.update_previous_solution()

        # Write the solution 
        stp.write(time_stamp=t)

        # Plot the numerical and exact solutions
        sol_exact = get_c_exact(x_plt, a, b, t, diffusivity, sigma_0)
        c_vals = stp.get_solution_function().x.array

        c_vals = stp.get_solution_function().x.array.copy()
        sol_exact = get_c_exact(x_plt, a, b, t, diffusivity, sigma_0)

        c_vals_list.append(c_vals)
        sol_exact_list.append(sol_exact)
        time_vals.append(t)

        # Step forward in time
        t += dt


    # Plot the evolution of the solution over time using matplotlib animation
    fig, ax = plt.subplots()
    line_num, = ax.plot([], [], label='Numerical')
    line_exact, = ax.plot([], [], label='Exact')
    ax.set_xlim(0, 12800)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

    def update(frame):
        line_num.set_data(x_plt, c_vals_list[frame])
        line_exact.set_data(x_plt, sol_exact_list[frame])
        ax.set_title(f"t = {time_vals[frame]:.0f}")
        return line_num, line_exact

    ani = animation.FuncAnimation(fig, update, frames=len(time_vals), blit=True, interval=100)
    ani.save('solution_evolution.mp4', writer='ffmpeg')
    plt.close(fig)

