Demo steady Stokes
====================

This demo demonstrates how to solve a steady-stokes flow problem.

The problem statement is as follows:

**Strong form:**

.. math:: 
	-\mu \nabla^2 \underline{u} + \nabla p - \underline{b} = \underline{0} \\
	\\
	\nabla \cdot \underline{u} = 0 \\

The body forces are given by:

.. math::
	\underline{b} = (b_1, b_2) \\
	\\
	b_1 = (12 - 24x_2)x_1^4 + (-24 + 48x_2)x_1^3 + &\\  
	    	(-48x_2 + 72x_2^2 - 48x_2^3 +12)x_1^2 + &\\
		(-2 + 24x_2 - 72x_2^2 + 48x_2^3)x_1 + 1 - &\\
		4x_2 +12x_2^2 - 8x_2^3 &\\
	b_2 = (8 - 48x_2 + 48x_2^2)x_1^3 + (-12 + 72x_2 - 72x_2^2)x_2^2 + &\\
		(4 - 24x_2 + 48x_2^2 - 48x_2^3 + 24x_2^4)x_1 - &\\
		12x_2^2 + 24x_2^3 - 12x_2^4 &\\
	\\
	u_1 = x_1^2 (1-x_1)^2 (2x_2 - 6x_2^2 + 4x_2^3) &\\
	u_2 = -x_2^2 (1- x_2)^2 (2x_1 - 6x_11^2 + 4x_1^3) &\\
	p = x_1(1-x_1) & \\

We first import relevant libraries and define a mesh. Here, we use fe.UnitSquareMesh to define
a 2D mesh from 0 to 1 with a size of 64x64. Then, we define feFlow's ``Mesh`` object.

.. code-block:: python
	 
	import numpy as np
	import matplotlib.pyplot as plt

	# ------------------------------------------------------- #

	import fenics as fe
	from feFlow.physics import IncompressibleStokes
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver

	# Define mesh
	ne = 64
	RM = fe.UnitSquareMesh(ne, ne)
	mesh = Mesh(mesh=RM)

Next, we define and mark the mesh boundaries. For simple domains we can use feFlow's 
``Mesh.mark_boundary`` method. Each time I mark the boundary, I supply a function handle
that take in the coordinate position ``x`` and any other arguments needed.

For example, the ``left`` function takes in ``x`` and the ``left_bnd`` value.
Calling ``mesh.mark_boundary(1, left, (0.))`` reads:
Set the boundary id to ``1`` for all points ``x`` such that ``left(x, 0.)`` returns ``True``

.. code-block:: python

	def left(x, left_bnd):
	    return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
	def right(x, right_bnd):
	    return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
	def top(x, top_bnd):
	    return abs(top_bnd - x[1]) < fe.DOLFIN_EPS
	def bottom(x, bottom_bnd):
	    return abs(x[1] - bottom_bnd) < fe.DOLFIN_EPS
	mesh.mark_boundary(1, left, (0.))
	mesh.mark_boundary(2, bottom, (0.))
	mesh.mark_boundary(3, right, (1.))
	mesh.mark_boundary(4, top, (1.))

Next, I define the relevant physics. In this case, it is the ``IncompressibleStokes`` physics. Then I set the finite element
to ``Continuous Galerkin of degree 1``. Finally, I set the function space of the problem.

code-block:: python

	ics = IncompressibleStokes(mesh)
	ics.set_element('CG', 1, 'CG', 1)
	ics.set_function_space()


Next, we set the coefficients on each term and define the body force vector. They are split into subcomponents for clarity. 

.. code-block:: python

	mu = 1.
	bx4 = "( 12 - 24*x[1]) * pow(x[0], 4)"
	bx3 = "(-24 + 48*x[1]) * pow(x[0], 3)"
	bx2 = "( 12 - 48*x[1] + 72*pow(x[1], 2) - 48*pow(x[1], 3) ) * pow(x[0], 2)"
	bx1 = "( -2 + 24*x[1] - 72*pow(x[1], 2) + 48*pow(x[1], 3) ) * x[0]"
	bx0 = "(  1 -  4*x[1] + 12*pow(x[1], 2) -  8*pow(x[1], 3) )"

	by3 = "(  8 - 48*x[1] + 48*pow(x[1], 2) ) * pow(x[0], 3)"
	by2 = "(-12 + 72*x[1] - 72*pow(x[1], 2) ) * pow(x[0], 2)"
	by1 = "(  4 - 24*x[1] + 48*pow(x[1], 2) - 48*pow(x[1], 3) + 24*pow(x[1], 4) ) * x[0]"
	by0 = "(              - 12*pow(x[1], 2) + 24*pow(x[1], 3) - 12*pow(x[1], 4) )"

	bx = "%s + %s + %s + %s + %s" % (bx4, bx3, bx2, bx1, bx0)
	by = "%s + %s + %s + %s" % (by3, by2, by1, by0)

Then, we define the body force we just defined to a FEniCS expression and set it and our constant mu. 

.. code-block:: python

	b = fe.Expression((bx, by), degree=4)
	ics.set_body_force(b)
	ics.set_dynamic_viscosity(mu)

Now we set the weak form and add stabilization:

.. code-block:: python

	ics.set_weak_form()
	ics.add_stab()

We then write a dictionary where the 'key' is the boundary id, and the 'value,' a dictionary 
indicating the type and value of the boundary conditions. The types can be either 
``dirichlet`` or ``neumann``. **See FEM theory for the difference.**

.. code-block:: python

	bc_dict = {1: {'type': 'dirichlet', 'field': 'velocity', 'value': 'zero'},
	           2: {'type': 'dirichlet', 'field': 'velocity', 'value': 'zero'},
	           3: {'type': 'dirichlet', 'field': 'velocity', 'value': 'zero'},
	           4: {'type': 'dirichlet', 'field': 'velocity', 'value': 'zero'}}
	ics.set_bcs(bc_dict)

Setting up the output:

.. code-block:: python
	
	ics.set_writer('output/benchmark_stokes.pvd')

We finalize the set-up with setting the type of problem (linear or nonlinear) and linear algebra 
solver. In this case, we 
have a linear PDE as our governing equation, so we set ``LinearProblem`` with the arguement as 
our physics class.

.. code-block:: python

	problem = LinearProblem(ics)
	la_solver = fe.LUSolver()
	solver = LinearSolver(mesh.comm, problem, la_solver)

Once the setup is complete, we can solve and write our solution out. 

.. code-block:: python

	solver.solve()
	ics.write()


**The full script:**

We have included the exact solution in the full script.

.. code-block:: python

	import numpy as np
	import matplotlib.pyplot as plt

	# ------------------------------------------------------- #

	import fenics as fe
	from feFlow.physics import IncompressibleStokes
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver

	# Define mesh
	ne = 64
	RM = fe.UnitSquareMesh(ne, ne)
	mesh = Mesh(mesh=RM)

	# Mark mesh
	def left(x, left_bnd):
	    return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
	def right(x, right_bnd):
	    return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
	def top(x, top_bnd):
	    return abs(top_bnd - x[1]) < fe.DOLFIN_EPS
	def bottom(x, bottom_bnd):
	    return abs(x[1] - bottom_bnd) < fe.DOLFIN_EPS
	mesh.mark_boundary(1, left, (0.))
	mesh.mark_boundary(2, bottom, (0.))
	mesh.mark_boundary(3, right, (1.))
	mesh.mark_boundary(4, top, (1.))

	# Define problem
	ics = IncompressibleStokes(mesh)
	ics.set_element('CG', 1, 'CG', 1)
	ics.set_function_space()

	# Set coefficients on each term and define the body force vector
	mu = 1.
	bx4 = "( 12 - 24*x[1]) * pow(x[0], 4)"
	bx3 = "(-24 + 48*x[1]) * pow(x[0], 3)"
	bx2 = "( 12 - 48*x[1] + 72*pow(x[1], 2) - 48*pow(x[1], 3) ) * pow(x[0], 2)"
	bx1 = "( -2 + 24*x[1] - 72*pow(x[1], 2) + 48*pow(x[1], 3) ) * x[0]"
	bx0 = "(  1 -  4*x[1] + 12*pow(x[1], 2) -  8*pow(x[1], 3) )"

	by3 = "(  8 - 48*x[1] + 48*pow(x[1], 2) ) * pow(x[0], 3)"
	by2 = "(-12 + 72*x[1] - 72*pow(x[1], 2) ) * pow(x[0], 2)"
	by1 = "(  4 - 24*x[1] + 48*pow(x[1], 2) - 48*pow(x[1], 3) + 24*pow(x[1], 4) ) * x[0]"
	by0 = "(              - 12*pow(x[1], 2) + 24*pow(x[1], 3) - 12*pow(x[1], 4) )"

	bx = "%s + %s + %s + %s + %s" % (bx4, bx3, bx2, bx1, bx0)
	by = "%s + %s + %s + %s" % (by3, by2, by1, by0)
	b = fe.Expression((bx, by), degree=4)
	ics.set_body_force(b)
	ics.set_dynamic_viscosity(mu)

	# Set weak form
	ics.set_weak_form()

	# Set stabilization
	ics.add_stab()

	# Set bc
	bc_dict = {1: {'type': 'dirichlet', 'field': 'velocity', 'value': 'zero'},
	           2: {'type': 'dirichlet', 'field': 'velocity', 'value': 'zero'},
	           3: {'type': 'dirichlet', 'field': 'velocity', 'value': 'zero'},
	           4: {'type': 'dirichlet', 'field': 'velocity', 'value': 'zero'}}
	ics.set_bcs(bc_dict)

	# Setup io
	ics.set_writer('output/benchmark_stokes.pvd')

	# Set problem
	problem = LinearProblem(ics)

	# Set solver
	la_solver = fe.LUSolver()
	solver = LinearSolver(mesh.comm, problem, la_solver)

	# Solve
	solver.solve()

	u0e = " pow(x[0], 2)*pow(1-x[0], 2)*(2*x[1] - 6*pow(x[1], 2) + 4*pow(x[1], 3))"
	u1e = "-pow(x[1], 2)*pow(1-x[1], 2)*(2*x[0] - 6*pow(x[0], 2) + 4*pow(x[0], 3))"
	u_exact = fe.Expression( (u0e, u1e), degree=2 )
	u_exact = fe.interpolate(u_exact, ics.V.sub(0).collapse())
	fe.File("output/ue.pvd") << u_exact
	ics.write()


