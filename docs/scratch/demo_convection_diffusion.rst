Demo convection-diffusion
===========================

This demo demonstrates how to solve a 1D transient convection-diffusion problem.

The following problem was taken from "Problem 1" from "Benchmarks for the Transport Equation:
The Convection_Diffusion Forum and Beyond" by Baptista and Adams, 1995

**Strong form**

.. math::

	\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2} - u \frac{\partial c}{\partial c} - f

The Gaussian Source Solution is given as:

.. math::

	c(x,t) = \frac{\sigma_0}{\sigma}exp(-\frac{(x-\bar{x})^2}{2\sigma^2} \\
	\\
	\sigma^2 = \sigma_0^2 +2Dt \\
	\\
	\bar{x} = x_0 + \int_0^t u(\tau) d\tau \\
	\\

We first import relevant libraries and define a mesh. Here, we use fe.InetervalMesh to create a mesh from 0 to 12800 with 128 elements.

.. code-block:: python

	import numpy as np
	import matplotlib.pyplot as plt
	
	import fenics as fe
	from feFlow.physics import TGScalarTransport
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver

	ne = 128
	IM = fe.IntervalMesh(ne,0,12800)
	mesh = Mesh(mesh=IM)

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
	mesh.mark_boundary(1,left,(0.))
	mesh.mark_boundary(2, right, (12800))

Next, we define several functions for x_bar, sigma, and the exact solution. We will use the exact solution to create the initial condition for our numerical setup.

.. code-block:: python

	def get_x_bar(a_a,a_b,a_t):
	    x_bar = (a_a - a_a * np.cos(a_b * a_t)) / a_b
	    return x_bar

	def get_sigma(a_sigma_0,a_D,a_t):
	    sigma = np.sqrt(a_sigma_0**2 + 2 * a_D * a_t)
	    return sigma

	def get_c_exact(a_x, a_a,a_b,a_t,a_D,a_sigma_0):
	    sigma = get_sigma(a_sigma_0, a_D, a_t)
	    x_bar = get_x_bar(a_a, a_b, a_t)
	    c = (a_sigma_0 / sigma) * np.exp(-1 * (a_x - x_bar)**2 / (2 * sigma**2))
	    return c

Next, I define the relevant physics. In this case, it is the ``ScalarTransport`` physics. Then I set the finite element
to ``Continuous Galerkin of degree 1``. Finally, I set the function space of the problem based on the finite element.

.. code-block:: python

	dt = 96
	t_0 = 3000
	st = TGScalarTransport(mesh, dt, theta=0.5)
	st.set_element('CG', 1)
	st.set_function_space()

Now, we need to set the coeffifients for each term. In this example, we will use D = 2.0. The 
scalar tranporst class allows these coefficients to be transient. So, we set both the current 
and updated values to a constant D.

.. code-block:: python

	D = 2.0
	D0 = fe.Expression('D', degree=1, D=D, t=0)
	Dn = fe.Expression('D', degree=1, D=D, t=0)
	st.set_diffusion_coefficient(D0,Dn)

We have a time-dependent function for velocity. So, we will create two separate functions u0  and un and update them with the appropriate t.

.. code-block:: python

	a = 1.5
	b = 2 * np.pi / 9600
	u0 = fe.Expression('a * sin(b*t)', degree=1, a=a, b=b, t=0)
	un = fe.Expression('a * sin(b*t)', degree=1, a=a, b=b, t=0)
	st.set_advection_velocity(u0, un)

We will do a similar process for the reaction term. However, since we are not including reactions in this problem, we will set f0 and fn to 0.

.. code-block:: python

	f0 = fe.Expression('0', degree=1, t=0)
	fn = fe.Expression('0', degree=1, t=0)
	st.set_reaction(f0, fn)

Next we set the weak formulation of the ADR equations. We have already set the coefficients terms 
and now will call ``set_weak_form()`` to complete setting the basic weak formulation.

.. code-block:: python

	st.set_weak_form()

Next, we set the boundary and initial conditions. We will use the exact solution to find our initial condition. 

.. code-block:: python

	sigma_0 = 264
	sigma = get_sigma(sigma_0, D, t_0)
	x_bar = get_x_bar(a, b, t_0)
	c0 = fe.interpolate(fe.Expression('s_0/s * exp(-1*pow(x[0]-x_bar,2)/(2*pow(s,2)))',
	                                  s_0=sigma_0, s=sigma, x_bar=x_bar, degree=1), st.V)
	st.set_initial_condition(c0)

We then write a dictionary where the 'key' is the boundary id, and the 'value,' a dictionary 
indicating the type and value of the boundary conditions. The types can be either 
``dirichlet`` or ``neumann``. **See FEM theory for the difference.**

.. code-block:: python

	bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.)},
	           2:{'type': 'dirichlet', 'value': fe.Constant(0.)}}
	st.set_bcs(bc_dict)

We finalize the set-up with setting the type of problem (linear or nonlinear) and linear algebra 
solver. In this case, we 
have a linear PDE as our governing equation, so we set ``LinearProblem`` with the arguement as 
our physics class. 

.. code-block:: python

	problem = LinearProblem(st)
	la_solver = fe.LUSolver()
	solver = LinearSolver(mesh.comm, problem, la_solver)

Finally, we can solve the time-marching problem at each step using a loop. We update 
the reaction term and current time at each time step and re-solve the problem. The solution 
at the current time step is stored as ``st.current_solution``.

.. code-block:: python

	t = t_0
	t_end = 7200
	while t <= t_end:	
		u0.t = t
		un.t = t + dt
		solver.solve()
		st.set_advection_velocity(u0, un)
		st.update_sol()
		t += dt


**The full script:**

We have included plotting the exact solution is this script.

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

	This demo demonstrates how to do a convection-diffusion problem using the transport class

	Author: njrovito
	'''

	# ------------------------------------------------------- #
	import numpy as np
	import matplotlib.pyplot as plt

	# ------------------------------------------------------- #

	import fenics as fe
	from feFlow.physics import TGScalarTransport
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver

	# ------------------------------------------------------- #

	# Define mesh
	ne = 128
	IM = fe.IntervalMesh(ne,0,12800)
	mesh = Mesh(mesh=IM)

	# Mark mesh
	def left(x, left_bnd):
	    return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
	def right(x, right_bnd):
	    return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
	mesh.mark_boundary(1,left,(0.))
	mesh.mark_boundary(2, right, (12800))

	# Defines x_bar
	def get_x_bar(a_a,a_b,a_t):
	    x_bar = (a_a - a_a * np.cos(a_b * a_t)) / a_b
	    return x_bar

	# Defines sigma
	def get_sigma(a_sigma_0,a_D,a_t):
	    sigma = np.sqrt(a_sigma_0**2 + 2 * a_D * a_t)
	    return sigma

	# Defines exact solution
	def get_c_exact(a_x, a_a,a_b,a_t,a_D,a_sigma_0):
	    sigma = get_sigma(a_sigma_0, a_D, a_t)
	    x_bar = get_x_bar(a_a, a_b, a_t)
	    c = (a_sigma_0 / sigma) * np.exp(-1 * (a_x - x_bar)**2 / (2 * sigma**2))
	    return c

	# Define problem
	dt = 96
	t_0 = 3000
	st = TGScalarTransport(mesh, dt, theta=0.5)
	st.set_element('CG', 1)
	st.set_function_space()

	# Diffusivity (here set as a constant)
	D = 2.0
	D0 = fe.Expression('D', degree=1, D=D, t=0)
	Dn = fe.Expression('D', degree=1, D=D, t=0)
	st.set_diffusion_coefficient(D0,Dn)

	# For the velocity term, we have a time-dependent velocity.
	# We will create two separate functions u0  and un and update
	# them with the appropriate t.
	a = 1.5
	b = 2 * np.pi / 9600
	u0 = fe.Expression('a * sin(b*t)', degree=1, a=a, b=b, t=0)
	un = fe.Expression('a * sin(b*t)', degree=1, a=a, b=b, t=0)
	st.set_advection_velocity(u0, un)

	# Similarly, we create f0 and fn for the reaction term (here set to zero)
	f0 = fe.Expression('0', degree=1, t=0)
	fn = fe.Expression('0', degree=1, t=0)
	st.set_reaction(f0, fn)

	# Set weak form
	st.set_weak_form()

	# Set initial condition
	x = np.linspace(0, 12800, ne, endpoint=True) # for plotting
	sigma_0 = 264
	sigma = get_sigma(sigma_0, D, t_0)
	x_bar = get_x_bar(a, b, t_0)
	c0 = fe.interpolate(fe.Expression('s_0/s * exp(-1*pow(x[0]-x_bar,2)/(2*pow(s,2)))',
	                                  s_0=sigma_0, s=sigma, x_bar=x_bar, degree=1), st.V)
	st.set_initial_condition(c0)

	# Set bc
	bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.)},
	           2:{'type': 'dirichlet', 'value': fe.Constant(0.)}}
	st.set_bcs(bc_dict)

	# Set problem
	problem = LinearProblem(st)

	# Set solver
	la_solver = fe.LUSolver()
	solver = LinearSolver(mesh.comm, problem, la_solver)

	# Begin transient section
	t = t_0
	t_end = 7200
	while t <= t_end:
	    # Update reaction term
	    # f0.t = t
	    # fn.t = t + dt
	    # Update diffusivity term
	    # D0.t = t
	    # Dn.t = t + dt
	    # Update velocity term
	    u0.t = t
	    un.t = t + dt

	    # Solve
	    solver.solve()
	    st.set_advection_velocity(u0, un)

	    # Update previous solution
	    st.update_sol()

	    # Update time
	    t += dt

	    # Plot computed solution against exact solution
	    sol_exact = get_c_exact(x, a, b, t, D, sigma_0)
	    fe.plot(st.current_sol, label='Computed solution')
	    plt.plot(x, sol_exact, 'r--', label='Exact solution')
	    plt.legend()
	    plt.title('t = %.4f' % t)
	    plt.ylim([-0.2, 2])
	    plt.pause(0.1)
	    plt.cla()








