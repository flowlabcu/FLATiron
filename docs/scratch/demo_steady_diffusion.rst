Demo steady diffusion
======================

This demo demonstrates how to solve a steady diffusion problem. 

Demo of 1D diffusion problem on an interval [0, 1].

The problem statement is as follows:

**Strong form: **

.. math::

	0 = \frac{ \partial ^ 2 c }{ \partial x^2 } + f &\\ 
	\\
	f = -\sin (\pi x) &\\
	\\
	c(0) = 1 & \\
	c'(0) = 1 & \\

The exact solution takes the following form:

.. math::
	
	c = 1 + x - \frac{x}{\pi} - \frac{ \sin (\pi x) }{ \pi^2 } & \\

We first import relevant libraries and define a mesh. Here, we use fe.InetervalMesh to define
a 1D mesh from 0 to 1 with 10 elements. Then, we define feFlow's ``Mesh`` object. 

.. code-block:: python

	import numpy as np
	import matplotlib.pyplot as plt
	import fenics as fe
	from feFlow.physics import ScalarTransport
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver
	
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

	mesh.mark_boundary(1, left, (0.))
	mesh.mark_boundary(2, right, (1.))

Next, I define the relevant physics. In this case, it is the ``ScalarTransport`` physics. Then I set the finite element
to ``Continuous Galerkin of degree 1``. Finally, I set the function space of the problem based on the finite element.

.. code-block:: python

	st = ScalarTransport(mesh)
	st.set_element('CG', 1)
	st.set_function_space()

Next, we set the constants. The scalar transport class supports advection-diffusion-reaction problems, so we set 
the advection velocity to zero. 

.. code-block:: python

	st.set_advection_velocity(0)
	st.set_diffusion_coefficient(1)
	st.set_reaction(fe.Expression("-sin(pi*x[0])", degree=1))

Next we set the weak formulation of the ADR equations. We have already set the coefficients terms 
and now will call ``set_weak_form()`` to complete setting the basic weak formulation.

.. code-block:: python

	st.set_weak_form()

We then write a dictionary where the 'key' is the boundary id, and the 'value,' a dictionary 
indicating the type and value of the boundary conditions. The types can be either 
``dirichlet`` or ``neumann``. **See FEM theory for the difference.**

.. code-block:: python

	bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(1.)},
           2:{'type': 'neumann', 'value': fe.Constant(1.)}}
	st.set_bcs(bc_dict)

We finalize the set-up with setting the type of problem (linear or nonlinear) and linear algebra 
solver. In this case, we 
have a linear PDE as our governing equation, so we set ``LinearProblem`` with the arguement as 
our physics class. 

.. code-block:: python

	problem = LinearProblem(st)
	la_solver = fe.LUSolver()
	solver = LinearSolver(mesh.comm, problem, la_solver)

finally, we solve the problem and write the solution.

.. code-block:: python

	solver.solve()
	st.write()

**Full script**

We have provided plotting of the full script in the folowing:

.. code-block:: python

	'''
	Demo for 1D diffusion equation on an interval [0,1]
	0 = d^2c/dx^2 + f
	f = -sin(pi*x)
	c[0] = 1
	c'[1] = 1
	exact solution: c = 1 + x - x/pi - sin(pi*x)/pi**2

	This demo demonstrates how to solve steady problem
	using the ScalarTransport class with both dirichlet and neumann boundary condition
	'''

	import numpy as np
	import matplotlib.pyplot as plt

	# ------------------------------------------------------- #

	import fenics as fe
	from feFlow.physics import ScalarTransport
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver

	# Define mesh
	ne = 100
	IM = fe.IntervalMesh(ne, 0, 1)
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
	st.set_advection_velocity(0)
	st.set_diffusion_coefficient(1)
	st.set_reaction(fe.Expression("-sin(pi*x[0])", degree=1))

	# Set weak form
	st.set_weak_form()

	# Set bc
	bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(1.)},
	           2:{'type': 'neumann', 'value': fe.Constant(1.)}}
	st.set_bcs(bc_dict)

	# Set problem
	problem = LinearProblem(st)

	# Set solver
	la_solver = fe.LUSolver()
	solver = LinearSolver(mesh.comm, problem, la_solver)

	# Set output file
	st.set_writer("u.pvd", "u")

	# Solve
	solver.solve()
	st.write()

	# Plot solution
	x = np.linspace(0, 1, ne+1)
	sol_exact = 1 + x - x/np.pi - np.sin(np.pi*x)/np.pi**2
	fe.plot(st.current_sol, label='Computed solution')
	plt.plot(x, sol_exact, 'r--', label='Exact solution')
	plt.legend()
	plt.show()

