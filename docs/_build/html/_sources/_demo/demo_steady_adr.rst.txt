======================================================
Demo: Steady Advection-Diffusion-Reaction Equation
======================================================

This demo code demonstrates how to solve a steady Advection-Diffusion-Reaction problem. This demo implementation can 
be found in **demo/demo_steady_adr/demo_steady_adr_1D.py**.

Problem Definition
--------------------

Strong form

.. math::

    u\frac{\partial c}{\partial x} - D\frac{\partial^2 c}{\partial x^2} = R \\

Where the reaction being a constant function:

.. math::

    R=1 \\

The boundary conditions are:

.. math::

    c(0)=1.0 \;,\; c(1)=0.5 \\

The analytical solution of this problem is:

.. math::

    c(x) = 1 - \frac{1}{2}x


Implementation
-----------------

We begin by importing the necessary libraries and creating a mesh.

.. code-block:: python 

    import matplotlib.pyplot as plt
    import numpy as np

    from flatiron_tk import constant
    from flatiron_tk.mesh import LineMesh
    from flatiron_tk.physics import SteadyScalarTransport
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

    # Create Mesh
    mesh = LineMesh(0, 1, 1/10)

Next, we define the relevant physics. In this case, it is the ``SteadyScalarTransport`` physics. Then we set the finite element
to ``Continuous Galerkin of degree 1``. Finally, we set the function space of the problem based on the finite element. Here, we also 
set the output writer to write the solution to an XDMF file.

.. code-block:: python 

    # Define Problem 
    stp = SteadyScalarTransport(mesh, tag='c')
    stp.set_writer('output', 'xdmf')
    stp.set_element('CG', 1)
    stp.build_function_space()

Next we set the weak formulation of the ADR equations. We do this by setting the advection/diffusion/reaction terms
and finally calling ``set_weak_form()`` to complete setting the basic weak formulation.

.. code-block:: python 

    # Set parameters
    stp.set_advection_velocity(0.0)
    stp.set_diffusivity(1.0)
    stp.set_reaction(0.0)

    # Set weak form and stabilization
    stp.set_weak_form()
    stp.add_stab() 

For a higher Peclet number problem, Galerkin finite element formulation is known to be unstable. 
Here, we provide SUPG stabilizationused for high Peclet number problems. Simply call ``add_stab()`` to add 
the stabilization to the weak formulation. The input ``su`` refer to the type of SUPG stabilization constant. 
See :doc:`/_modules/physics_modules/module_scalar_transport` for different choices of stabilization constants.

Next, we define the boundary conditions. We take in a dictionary with the key being the boundary id, and the value
being a dictionary indicating the type and value of the boundary conditions. The types can be either ``dirichlet`` or ``neumann``. The `constant`
function is a helper function that wraps the dolfinx constant object. 

.. code-block:: python 

    # Define Boundary Conditions
    left_bc = constant(mesh, 1.0)
    right_bc = constant(mesh, 0.5)

    bc_dict = {
        1: {'type': 'dirichlet', 'value': left_bc},
        2: {'type': 'neumann', 'value': right_bc}
    }

    stp.set_bcs(bc_dict)

Next, we set up the physics solver and solve the problem.

.. code-block:: python 

    # Define and Solve Problem
    problem = NonLinearProblem(stp)
    solver = NonLinearSolver(mesh.msh.comm, problem)

    solver.solve()
    stp.write()

Finally, we plot the solution:

.. code-block:: python 

    # Extract solution for plotting
    x = stp.mesh.msh.geometry.x[:, 0]  # Assuming 1D mesh, extract x-coordinates
    u = stp.solution.x.array           # Solution values as a NumPy array

    # Sort points for plotting (since mesh nodes may be unordered)
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    u_sorted = u[sorted_indices]

    # Plot
    plt.plot(x_sorted, u_sorted, marker='o', label="Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("flatiron_tk 1D Steady Scalar Transport Solution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("steady_scalar_transport_solution.png", dpi=300)
    plt.show()

Full Script
----------------

.. code-block:: python 

    import matplotlib.pyplot as plt
    import numpy as np

    from flatiron_tk import constant
    from flatiron_tk.mesh import LineMesh
    from flatiron_tk.physics import SteadyScalarTransport
    from flatiron_tk.solver import NonLinearProblem
    from flatiron_tk.solver import NonLinearSolver

    # Create Mesh
    mesh = LineMesh(0, 1, 1/10)

    # Define Problem 
    stp = SteadyScalarTransport(mesh, tag='c')
    stp.set_writer('output', 'xdmf')
    stp.set_element('CG', 1)
    stp.build_function_space()

    # Set parameters
    stp.set_advection_velocity(0.0)
    stp.set_diffusivity(1.0)
    stp.set_reaction(0.0)

    # Set weak form and stabilization
    stp.set_weak_form()
    stp.add_stab()  

    # Define Boundary Conditions
    left_bc = constant(mesh, 1.0)
    right_bc = constant(mesh, 0.5)

    bc_dict = {
        1: {'type': 'dirichlet', 'value': left_bc},
        2: {'type': 'neumann', 'value': right_bc}
    }

    stp.set_bcs(bc_dict)

    # Define and Solve Problem
    problem = NonLinearProblem(stp)
    solver = NonLinearSolver(mesh.msh.comm, problem)

    solver.solve()
    stp.write()

    # Extract solution for plotting
    x = stp.mesh.msh.geometry.x[:, 0]  # Assuming 1D mesh, extract x-coordinates
    u = stp.solution.x.array           # Solution values as a NumPy array

    # Sort points for plotting (since mesh nodes may be unordered)
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    u_sorted = u[sorted_indices]

    # Plot
    plt.plot(x_sorted, u_sorted, marker='o', label="Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("flatiron_tk 1D Steady Scalar Transport Solution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("steady_scalar_transport_solution.png", dpi=300)
    plt.show()

