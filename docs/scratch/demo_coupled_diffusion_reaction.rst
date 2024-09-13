Coupled diffusion-reaction problem with surface reaction
==============================================================

This demo code demonstrate how to solve a steady coupled Diffusion-Reaction problem with surface reaction terms at the boundary. This demo is used to demonstrate how to use the feFlow MultiPhysics module

.. math::

    \\
    \text{Define conecntration of chemical species $A, B, C$} \\
    \text{For the domain $x \in [0, L]$} \\

    \text{Strong form:} \;\;
    D_A \frac{d^2A}{dx^2} - k_v A B = 0 \\
    D_B \frac{d^2B}{dx^2} - 2k_v A B = 0 \\
    D_C \frac{d^2C}{dx^2} + k_v A B = 0 \\

    \\

    \text{Where} \;\;
    D_A,D_B,D_C \; \text{are diffusivity for each species and} \\
    k_v,k_s \; \text{are volumetric and surface reaction rates respectively} \\

    \\

    \text{BC:} \;\;
    A(x=0) = C0 \\
    B(x=0) = C0 \\
    C(x=0) = 0 \\
    \frac{dA}{dx}(x=L) = - \frac{k_s}{D_A} A B \\
    \frac{dB}{dx}(x=L) = - \frac{2k_s}{D_B} A B \\
    \frac{dC}{dx}(x=L) = \frac{k_s}{D_C} A B \\



Fist, we import code the relevant modules from feFlow and the basic libraries and define the mesh and constants

.. code-block:: python

    import fenics as fe
    from feFlow.physics import MultiPhysicsProblem, ScalarTransport
    from feFlow.mesh import Mesh
    from feFlow.problem import NonLinearProblem, NonLinearSolver

    # Define mesh
    ne = 10
    IM = fe.IntervalMesh(ne, 0, 1)
    h = 1/ne
    mesh = Mesh(mesh=IM)

    # Define constants
    D_A = 1.0; D_B = 1.0; D_C = 1.0 # diffusion coefficients
    k_v = 1 # Volumetric reaction rate
    k_s = 1 # Surface reaction rate
    C0 = 1 # Left BC for species A and B
    u = 0 # No advection

    # Mark mesh
    def left(x, left_bnd):
        return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
    def right(x, right_bnd):
        return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
    mesh.mark_boundary(1, left, (0.))
    mesh.mark_boundary(2, right, (1.))


Next I define the ``ScalarTransport`` problems for species A, B, and C

.. code-block:: python

    # Define the problem for species A
    A_pde = ScalarTransport(mesh, tag='A')
    A_pde.set_element('CG', 1)
    A_pde.set_advection_velocity(u)
    A_pde.set_diffusion_coefficient(D_A)

    # Define the problem for species B
    B_pde = ScalarTransport(mesh, tag='B')
    B_pde.set_element('CG', 1)
    B_pde.set_advection_velocity(u)
    B_pde.set_diffusion_coefficient(D_B)

    # Define the problem for species C
    C_pde = ScalarTransport(mesh, tag='C')
    C_pde.set_element('CG', 1)
    C_pde.set_advection_velocity(u)
    C_pde.set_diffusion_coefficient(D_C)


Now we set a ``MultiPhysicsProblem`` based on the three ``ScalarTransport`` problems that we created

.. code-block:: python

    coupled_physics = MultiPhysicsProblem(A_pde, B_pde, C_pde)
    coupled_physics.set_element()
    coupled_physics.set_function_space()


Set the coupling part of the equations here, we can see the coupling as the reaction terms we use the solution_function instead of trial function because this will be a nonlinear problem, and we will solve the problem using Newton iteration by taking the Gateaux derivative of the weak form W.R.T the solution functions. Finally, we set the weak formulation of the coupled physics by setting the linearity to ``False``.

.. code-block:: python

    A = coupled_physics.solution_function('A')
    B = coupled_physics.solution_function('B')
    C = coupled_physics.solution_function('C')
    A_pde.set_reaction(-k_v*A*B)
    B_pde.set_reaction(-2*k_v*A*B)
    C_pde.set_reaction(k_v*A*B)

    coupled_physics.set_weak_form(is_linear=False)


Now we set the boundary conditions dictionary foe each physics, and create an overall dictionary with the species tag called ``bc_dict`` which we supply into the ``coupled_physics`` object.

.. code-block:: python


    # Set BCs for specific physics
    A_bcs = {
            1: {'type': 'dirichlet', 'value': fe.Constant(C0)},
            2: {'type': 'neumann', 'value': -k_s*A*B/D_A}
            }

    B_bcs = {
            1: {'type': 'dirichlet', 'value': fe.Constant(C0)},
            2: {'type': 'neumann', 'value': -2*k_s*A*B/D_B}
            }

    C_bcs = {
            1: {'type': 'dirichlet', 'value': fe.Constant(0)},
            2: {'type': 'neumann', 'value': k_s*A*B/D_C}
            }

    bc_dict = {
            'A': A_bcs,
            'B': B_bcs,
            'C': C_bcs
              }
    coupled_physics.set_bcs(bc_dict)


Finally we solve with the nonlinear problem/solver and save the result

.. code-block:: python

    # Solve this problem using a nonlinear solver
    la_solver = fe.LUSolver()
    problem = NonLinearProblem(coupled_physics)
    solver = NonLinearSolver(mesh.comm, problem, la_solver)
    solver.solve()

    # Write solution
    coupled_physics.set_writer("output", "pvd")
    coupled_physics.write()

    # Plot solution
    solutions = coupled_physics.current_sol.split(deepcopy=True)
    fe.plot(solutions[0], label='A')
    fe.plot(solutions[1], label='B')
    fe.plot(solutions[2], label='C')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.show()




The code should give the following result

.. image:: ../../demo/coupled_diffusion_reaction/coupled_diffusion_reaction.png

**Here is the full script**

.. code-block:: python
    :linenos:

    import fenics as fe
    from feFlow.physics import MultiPhysicsProblem, ScalarTransport
    from feFlow.mesh import Mesh
    from feFlow.problem import NonLinearProblem, NonLinearSolver

    # Define mesh
    ne = 10
    IM = fe.IntervalMesh(ne, 0, 1)
    h = 1/ne
    mesh = Mesh(mesh=IM)

    # Define constants
    D_A = 1.0; D_B = 1.0; D_C = 1.0 # diffusion coefficients
    k_v = 1 # Volumetric reaction rate
    k_s = 1 # Surface reaction rate
    C0 = 1 # Left BC for species A and B
    u = 0 # No advection

    # Mark mesh
    def left(x, left_bnd):
        return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
    def right(x, right_bnd):
        return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
    mesh.mark_boundary(1, left, (0.))
    mesh.mark_boundary(2, right, (1.))

    # Define the problem for species A
    A_pde = ScalarTransport(mesh, tag='A')
    A_pde.set_element('CG', 1)
    A_pde.set_advection_velocity(u)
    A_pde.set_diffusion_coefficient(D_A)

    # Define the problem for species B
    B_pde = ScalarTransport(mesh, tag='B')
    B_pde.set_element('CG', 1)
    B_pde.set_advection_velocity(u)
    B_pde.set_diffusion_coefficient(D_B)

    # Define the problem for species C
    C_pde = ScalarTransport(mesh, tag='C')
    C_pde.set_element('CG', 1)
    C_pde.set_advection_velocity(u)
    C_pde.set_diffusion_coefficient(D_C)

    # Define a multiphysics problem as a combination of physics of
    # species A, B, C
    coupled_physics = MultiPhysicsProblem(A_pde, B_pde, C_pde)
    coupled_physics.set_element()
    coupled_physics.set_function_space()

    # Set the coupling part of the equations
    # here, we can see the coupling as the reaction terms
    # we use the solution_function instead of trial function because this will be a
    # nonlinear problem, and we will solve the problem using Newton iteration by taking
    # the Gateaux derivative of the weak form W.R.T the solution functions
    A = coupled_physics.solution_function('A')
    B = coupled_physics.solution_function('B')
    C = coupled_physics.solution_function('C')
    A_pde.set_reaction(-k_v*A*B)
    B_pde.set_reaction(-2*k_v*A*B)
    C_pde.set_reaction(k_v*A*B)

    # Set weakform. Make sure that the problem linearity
    # is set to False as this is a non-linear problem
    coupled_physics.set_weak_form(is_linear=False)

    # Set BCs for specific physics
    A_bcs = {
            1: {'type': 'dirichlet', 'value': fe.Constant(C0)},
            2: {'type': 'neumann', 'value': -k_s*A*B/D_A}
            }

    B_bcs = {
            1: {'type': 'dirichlet', 'value': fe.Constant(C0)},
            2: {'type': 'neumann', 'value': -2*k_s*A*B/D_B}
            }

    C_bcs = {
            1: {'type': 'dirichlet', 'value': fe.Constant(0)},
            2: {'type': 'neumann', 'value': k_s*A*B/D_C}
            }

    bc_dict = {
            'A': A_bcs,
            'B': B_bcs,
            'C': C_bcs
              }
    coupled_physics.set_bcs(bc_dict)

    # Solve this problem using a nonlinear solver
    la_solver = fe.LUSolver()
    problem = NonLinearProblem(coupled_physics)
    solver = NonLinearSolver(mesh.comm, problem, la_solver)
    solver.solve()

    # Write solution
    coupled_physics.set_writer("output", "pvd")
    coupled_physics.write()

    # Plot solution
    solutions = coupled_physics.current_sol.split(deepcopy=True)
    fe.plot(solutions[0], label='A')
    fe.plot(solutions[1], label='B')
    fe.plot(solutions[2], label='C')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.show()





