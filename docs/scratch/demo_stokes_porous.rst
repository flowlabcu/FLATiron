Demo pressure driven stokes flow through porous media
========================================================

This demo code demonstrates how to use the stokes flow solver for a pressure-driven flow problem
through an explicitly meshed porous media domain with a PSPG stabilized Q1P1 element

.. math::

    \\

    \text{Strong form:} \;\;
    \nabla p = \mu\nabla^2 u \\
    \nabla \cdot u = 0 \\

    \\

    \text{Stabilized weak form:} \;\;
    \mu\left( \nabla w, \nabla u \right) - \left(\nabla \cdot w, \right) + \left(q, \nabla \cdot u \right) + \left( P, \tau R\right)\\
    \text{Where:} \\
    \tau = \frac{h^2}{12\mu} \\
    \text{and}\\
    \text{R is the residue of the strong form} \\

    \\

    \text{BCs:} \;\;
    P_{inlet} = 1 \\
    P_{outlet} = 0 \\

This demo involves using the the standard stokes solver physics with PSPG stabilization. First we load the relevant libraries
and initialize the ``IncompressibleStokes`` solver with a Q1P1 element

.. code-block:: python
    import numpy as np
    import matplotlib.pyplot as plt
    import fenics as fe
    from feFlow.physics import IncompressibleStokes
    from feFlow.mesh import Mesh
    from feFlow.problem import LinearProblem, LinearSolver

    # Define mesh
    mesh = Mesh(mesh_file='porous.h5')

    # Define problem
    # element inputs are:
    # velocity_element_family, velocity_element_degree, pressure_element_family, pressure_element_degree
    ics = IncompressibleStokes(mesh)
    ics.set_element('CG', 1, 'CG', 1)
    ics.set_function_space()


Next we set the body forces and other constant properties, and generate the stabilized weakform

.. code-block:: python
    # Set coefficients on each term
    b = fe.Constant((0, 0))
    ics.set_body_force(b)
    ics.set_dynamic_viscosity(1)

    # Set weak form
    ics.set_weak_form()

    # Set stabilization
    ics.add_stab()


Now we define the boundary conditions using a pressure BC in the IncompressibleStokes physics

.. code-block:: python
    # Set Boundary conditions
    # Here the velocity conditions are explicitly set via a dirichlet BC
    # while the pressure bc is set via a neumann traction term
    # We are assuming that pressure >> grad(u) in the traction term
    p_inlet = 1.
    p_outlet = 0.
    bc_dict = {1: {'field': 'pressure', 'value': p_inlet},
               2: {'field': 'velocity', 'value': 'zero'},
               3: {'field': 'pressure', 'value': p_outlet},
               4: {'field': 'velocity', 'value': 'zero'},
               5: {'field': 'velocity', 'value': 'zero'}}
    ics.set_bcs(bc_dict)


Finally we setup the solver, and solve the problem. Note that we're doing a simple monolithic solve through a direct solver ``fe.LUSolver()`` because the problem is small and simple enough

 .. code-block:: python
    # Setup io
    ics.set_writer('output/porous.pvd')

    # Set problem
    problem = LinearProblem(ics)

    # Set solver
    la_solver = fe.LUSolver()
    solver = LinearSolver(mesh.comm, problem, la_solver)

    # Solve
    solver.solve()
    ics.write()


