Demo moving front
==================

This demo demonstrates how to solve a 2D pure convection problem.

The problem statement is as follows:

**Strong form:**

.. math::

	\frac{\partial c}{\partial t} + \underline{u} \cdot  \nabla c = D \nabla^2 c] \\
	\underline{u} = (0, 1) \\
	c(x, 0) = 0 \\
	x(0, t) = 1 \\

We first import relevant libraries and define a mesh. Here, we use fe.RectangMesh to define
a 2D mesh with 0 < x < L and 0 < y < L/10 with 100 elements in the x direction and 10 elements in the y direction. Then, we define feFlow's ``Mesh`` object. 

.. code-block:: python

	import numpy as np
	import matplotlib.pyplot as plt
	import fenics as fe
	from feFlow.physics import TGScalarTransport 
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver

	ne = 100
	L = 1
	RM = fe.RectangleMesh(fe.Point(0,0), fe.Point(L, L/10.), ne, int(ne/10), 'crossed')
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
	mesh.mark_boundary(3, right, (L))
	mesh.mark_boundary(4, top, (1.))	

Next, I define the relevant physics. In this case, it is the ``ScalarTransport`` physics. Then I set the finite element
to ``Continuous Galerkin of degree 1``. Finally, I set the function space of the problem based on the finite element

.. code-block:: python

	dt = 1e-3
	st = TGScalarTransport(mesh, dt, theta=0.5)
	st.set_element('CG', 1)
	st.set_function_space()

Now, we need to set the coeffifients for each term. In this example, we will use D = 1e-3. The 
scalar tranporst class allows these coefficients to be transient. So, we set both the current 
and updated values to a constant D. We also set the advection velocity with the constant velocity vector, and the reaction term to 0.

.. code-block:: python

	D = 1e-3
	u = fe.Constant((1, 0))
	st.set_advection_velocity(u, u)
	st.set_diffusion_coefficient(D, D)
	st.set_reaction(0, 0)

Next we set the weak formulation of the ADR equations. We have already set the coefficients terms 
and now will call ``set_weak_form()`` and add stabilization to complete setting the basic weak formulation.

.. code-block:: python

	st.set_weak_form()
	st.add_stab()

We then write a dictionary where the 'key' is the boundary id, and the 'value,' a dictionary 
indicating the type and value of the boundary conditions. The types can be either 
``dirichlet`` or ``neumann``. **See FEM theory for the difference.**

.. code-block:: python

	bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(1.)}}
	st.set_bcs(bc_dict)

We finalize the set-up with setting the type of problem (linear or nonlinear) and linear algebra 
solver. In this case, we 
have a linear PDE as our governing equation, so we set ``LinearProblem`` with the arguement as 
our physics class. 

.. code-block:: python

	problem = LinearProblem(st)
	la_solver = fe.LUSolver()
	solver = LinearSolver(mesh.comm, problem, la_solver)

Next, we set the output writer.

.. code-block:: python

	st.set_writer("u.h5", "u")

Finally, we can solve the time-marching problem at each step using a loop. We update 
the current time at each time step and re-solve the problem. The solution 
at the current time step is stored as ``st.current_solution``.

.. code-block:: python

	t = 0
	t_end = 0.5
	i = 0

	while t <= t_end:

		t = t + dt
		solver.solve()
		st.write(time_stamp=t)
		st.update_sol()

We have included plotting the exact solution is this script.

.. code-block:: python

	'''
	Demo for 2D transient pure convection problem
	dc/dt + u.grad(c) = D*laplace(c)
	u = (1, 0)
	c[t=0] = 0
	c[x=0] = 1

	This demo demonstrate 2D transient problem
	'''

	import numpy as np
	import matplotlib.pyplot as plt

	# ------------------------------------------------------- #

	import fenics as fe
	from feFlow.physics import TGScalarTransport 
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver

	# Define mesh
	ne = 100
	L = 1
	RM = fe.RectangleMesh(fe.Point(0,0), fe.Point(L, L/10.), ne, int(ne/10), 'crossed')
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
	mesh.mark_boundary(3, right, (L))
	mesh.mark_boundary(4, top, (1.))


	# Define problem
	dt = 1e-3
	st = TGScalarTransport(mesh, dt, theta=0.5)
	st.set_element('CG', 1)
	st.set_function_space()

	# Set coefficients on each term
	# here since we are in transient mode, we have to set
	# the function defining the previous and current time step.
	# Since D and u are constants, we repeat the values for both entry
	D = 1e-3
	u = fe.Constant((1, 0))
	st.set_advection_velocity(u, u)
	st.set_diffusion_coefficient(D, D)
	st.set_reaction(0, 0)

	# Set weak form
	st.set_weak_form()

	# Not setting the initial condition here means that
	# we are using a zero initial condition

	# Set stabilization
	st.add_stab()

	# Set bc
	bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(1.)}}
	st.set_bcs(bc_dict)

	# Set problem
	problem = LinearProblem(st)

	# Set solver
	la_solver = fe.LUSolver()
	solver = LinearSolver(mesh.comm, problem, la_solver)

	# Set IO
	st.set_writer("u.h5", "u")

	# Set time steps
	t = 0
	t_end = 0.5
	i = 0

	# Solve
	while t <= t_end:

	    # Update time
	    t = t + dt

	    # Solve
	    solver.solve()

	    # Save solution
	    st.write(time_stamp=t)

	    # Update previous solution
	    st.update_sol()

	# Define exact solution for benchmark
	x_mesh = np.linspace(0, 1, ne+1)
	def c_exact(x, t):
	    return np.heaviside(t-x, 0.5)

	# Extract centerline
	def centerline(c):
	    c_center = []
	    for xi in x_mesh:
	        p = fe.Point(xi, L/10/2)
	        c_center.append(c(p))
	    return np.array(c_center)

	# Plot field solution
	plt.figure()
	fe.plot(st.current_sol)

	# Plot centerline solution
	plt.figure()
	x = np.linspace(0, 1, ne+1)
	H = np.heaviside(0.5-x, 0.5)
	usol = st.current_sol
	uu = []
	for i in range(len(x)):
	    uu.append(usol(fe.Point(x[i], 0.05)))
	plt.plot(x, H, 'k--', label='Exact solution')
	plt.plot(x, uu, label='Computed solution')
	plt.legend()
	plt.grid(True)
	plt.show()










