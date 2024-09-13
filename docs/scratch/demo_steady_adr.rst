Steady state ADR
===================================================

This demo code demonstrate how to solve a steady Advection-Diffusion-Reaction problem

Demo for 1D advection-diffusion-reaction equation [0,1]
This problem recreates fig 2.1 in Donea's book: Finite Element Methods for Flow Problems. **DEVNOTE: Cite Donea's book**

The problem statement is as follows: **DEVNOTE: Figure out how to justify the equations to the left. This is really ugly right now**

.. math::

    \\

    \text{Strong form:} \;\;
    u\frac{\partial c}{\partial x} - D\frac{\partial^2 c}{\partial x^2} = R \\

    \\

    \text{With constant reaction:} \;\;
    R=1 \\

    \\

    \text{BC:} \;\;
    c(0)=0 \;,\; c(1)=0 \\

    \\

    \text{Analytical solution:} \;\;
    c = \frac{1}{u} \left( x - \frac{1-e^{ux/D}}{1-e^{u/D}} \right) \\



Fist, we import code the relevant modules from feFlow and the basic libraries
and, we define the mesh. Here we define 1D mesh (IntervalMesh) from 0 to 1 with 10 elements
then we initialize feFlow's mesh object.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import fenics as fe
    from feFlow.physics import ScalarTransport
    from feFlow.mesh import Mesh
    from feFlow.problem import LinearProblem, LinearSolver

    # Define mesh
    ne = 10
    IM = fe.IntervalMesh(ne, 0, 1)
    h = 1/ne
    mesh = Mesh(mesh=IM)


Next, we mark the boundary of the mesh. Since this is a simple domain, we can do it directly
from feFlow's Mesh mark_boundary() method. Each time I mark the boundary, I supply a function handle
that take in the coordinate position ``x`` and any other arguments needed.

For example, the ``left`` function takes in ``x`` and the ``left_bnd`` value.
Calling ``mesh.mark_boundary(1, left, (0.))`` reads:
Set the boundary id to ``1`` for all points ``x`` such that ``left(x, 0.)`` returns ``True``

.. code-block:: python

    def left(x, left_bnd):
        return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
    def right(x, right_bnd):
        return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
    mesh.mark_boundary(1, left, (0.))
    mesh.mark_boundary(2, right, (1.))

Next, I define the relevant physics. In this case, it is the ``ScalarTransport`` physics. Then I set the finite element
to ``Continuous Galerkin of degree 1``. Finally, I set the function space of the problem based on the finite element

.. code-block:: python

    # Define problem
    st = ScalarTransport(mesh)
    st.set_element('CG', 1)
    st.set_function_space()

Next we set the weak formulation of the ADR equations. We do this by setting the advection/diffusion/reaction terms
and finally calling ``set_weak_form()`` to complete setting the basic weak formulation.

For a higher Peclet number problem, Galerkin finite element formulation is known to be unstable. ``feFlow`` provides basic stabilization
techniques for these problem. Here we simply call ``add_stab()`` to add the stabilization to the weakform. The input ``su`` refer to the type
of stabilization constant. See the specific physics documentation for details. **DEVNOTE: Add link to the physics doc, and some points on the stab method**

.. code-block:: python

    # Set constants
    u = 1
    Pe = 5
    D = u/Pe/2*h
    R = 1.
    st.set_advection_velocity(u)
    st.set_diffusion_coefficient(D)
    st.set_reaction(R)

    # Set weak form
    st.set_weak_form()

    # Add supg term
    st.add_stab('su')


Next, we define the boundary conditions. We take in a dictionary with the key being the boundary id, and the value
being a dictionary indicating the type and value of the boundary conditions. The types can be either ``dirichlet`` or ``neumann``.
**See FEM theory for the difference**

.. code-block:: python

    # Set bc
    bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.)},
               2:{'type': 'dirichlet', 'value': fe.Constant(0.)}}
    st.set_bcs(bc_dict)

Finally, we set up a problem and solve. The final solution is encoded in the ``physics.current_sol`` variable

.. code-block:: python

    # Set problem
    problem = LinearProblem(st)

    # Set solver
    la_solver = fe.LUSolver()
    solver = LinearSolver(mesh.comm, problem, la_solver)
    solver.solve()

    # Plot solution
    x = np.linspace(0, 1, 100*(ne+1))
    g = u/D
    sol_exact = 1/u * (x - (1-np.exp(g*x))/(1-np.exp(g)))
    fe.plot(st.current_sol, linestyle='-', marker='o', label='Computed solution')
    plt.plot(x, sol_exact, 'r--', label='Exact solution')
    plt.grid(True)
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.xlabel('x')
    plt.ylabel('c')
    plt.legend()
    plt.savefig('demo_steady_adr.png')
    plt.show()

The code should give the following result

.. image:: ../../demo/steady_adr/demo_steady_adr.png 

**Here is the full script**

.. code-block:: python
    :linenos:

    import numpy as np
    import matplotlib.pyplot as plt
    import fenics as fe
    from feFlow.physics import ScalarTransport
    from feFlow.mesh import Mesh
    from feFlow.problem import LinearProblem, LinearSolver

    # Define mesh
    ne = 10
    IM = fe.IntervalMesh(ne, 0, 1)
    h = 1/ne
    mesh = Mesh(mesh=IM)

    # Mark mesh
    def left(x, left_bnd):
        return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
    def right(x, right_bnd):
        return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
    mesh.mark_boundary(1, left, (0.))
    mesh.mark_boundary(2, right, (1.))

    # Define problem
    st = ScalarTransport(mesh)
    st.set_element('CG', 1)
    st.set_function_space()

    # Set constants
    u = 1
    Pe = 5
    D = u/Pe/2*h
    R = 1.
    st.set_advection_velocity(u)
    st.set_diffusion_coefficient(D)
    st.set_reaction(R)

    # Set weak form
    st.set_weak_form()

    # Add supg term
    st.add_stab('su')

    # Set bc
    bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.)},
               2:{'type': 'dirichlet', 'value': fe.Constant(0.)}}
    st.set_bcs(bc_dict)

    # Set problem
    problem = LinearProblem(st)

    # Set solver
    la_solver = fe.LUSolver()
    solver = LinearSolver(mesh.comm, problem, la_solver)
    solver.solve()

    # Plot solution
    x = np.linspace(0, 1, 100*(ne+1))
    g = u/D
    sol_exact = 1/u * (x - (1-np.exp(g*x))/(1-np.exp(g)))
    fe.plot(st.current_sol, linestyle='-', marker='o', label='Computed solution')
    plt.plot(x, sol_exact, 'r--', label='Exact solution')
    plt.grid(True)
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.xlabel('x')
    plt.ylabel('c')
    plt.legend()
    plt.savefig('demo_steady_adr.png')
    plt.show()
