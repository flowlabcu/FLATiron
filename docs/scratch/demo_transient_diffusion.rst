Demo transient diffusion
==========================

This demo demonstrates how to solve a transient diffusion problem. 

Demo of 1D diffusion problem on an interval [0, 1].

The problem statement is as follows:

**Strong form:**

.. math ::
	
	\frac{\partial c}{\partial t} = D \frac{\partial^2 c }{\partial t^2 } + f
	\\
	f = t \sin (\pi x )
	\\

**With boundary conditions:**

.. math ::
	\frac{\partial c(0, t) }{\partial x } = 0
	\\
	\frac{\partial c(1, t) }{\partial x } = 0
	\\

**And intitial condition:**

.. math ::
	 c(x, 0) = \sin (\pi x)

**Exact solution:**

.. math ::
	c(x, t) = \alpha \gamma \sin(\pi x)
	\\
	\alpha  = \frac{\exp (-2 D \pi t)}{D^2 \pi ^4}
	\\
	\gamma  = 1 + D^2 \pi^4 + (D \pi ^ 2 t - 1)\exp (D \pi ^2 t)	
	\\

**devnote** Why are they right aligned now???

We first import relevant libraries and define a mesh. Here, we use fe.InetervalMesh to define
a 1D mesh from 0 to 1 with 10 elements. Then, we define feFlow's ``Mesh`` object. 

.. code-block:: python
	
	import numpy as np
	import matplotlib.pyplot as plt

	# ------------------------------------------------------- #

	import fenics as fe
	from feFlow.physics import TGScalarTransport 
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver

	# Define mesh
	ne = 10
	IM = fe.IntervalMesh(ne, 0, 1)
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

	mesh.mark_boundary(1, left, (0.))
	mesh.mark_boundary(2, right, (1.))

Next, I define the relevant physics. In this case, it is the ``ScalarTransport`` physics. Then I set the finite element
to ``Continuous Galerkin of degree 1``. Finally, I set the function space of the problem based on the finite element

.. code-block:: python
	
	dt = 1e-3
	st = TGScalarTransport(mesh, dt, theta=0.5)
	st.set_element('CG', 1)
	st.set_function_space()

Now, we need to set the coeffifients for each term. In this example, we will use D = 0.2. The 
scalar tranporst class allows these coefficients to be transient. So, we set both the current 
and updated values to a constant D.

The scalar transport class is also used for advection-diffusion problems. Since this example 
is only a transient diffusion problem, we will set the velocity values to zero. 

.. code-block:: python

	D = 0.2
	st.set_advection_velocity(0, 0)
	st.set_diffusion_coefficient(D, D)

For the reaction term, we have a function that is time dependent. We create two separate Fenics 
functions f0 and fn that we can update during the time-marching section of the code with the appropriate 
time value.  **Note that** ``fe.Expression`` **must be written as though they are in c.** 

.. code-block:: python

	f0 = fe.Expression("t*sin(pi*x[0])", degree=1, t=0)
	fn = fe.Expression("t*sin(pi*x[0])", degree=1, t=0)
	st.set_reaction(f0, fn)

Next we set the weak formulation of the ADR equations. We have already set the coefficients terms 
and now will call ``set_weak_form()`` to complete setting the basic weak formulation.

.. code-block:: python

	st.set_weak_form()

Next, we define the boundary and initial conditions. The initial condion is set by inerpolating 
a Fenics function. 

.. code-block:: python

	c0 = fe.interpolate(fe.Expression("sin(pi*x[0])", degree=1), st.V)
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

	t = 0
	t_end = 0.2
	i = 0
	
	while t <= t_end:

		# Update reaction term
    		f0.t = t
    		fn.t = t + dt
		
		# Update time
    		t = t + dt

    		# Solve
    		solver.solve()

    		# Update previous solution
    		st.update_sol()

		# Plot solution
		fe.plot(st.current_sol, label='Computed solution')
		plt.legend()
		plt.title('t = %.4f' % t)
		plt.ylim([0,1])
		plt.pause(0.1)
		plt.cla()

**DEVNOTE insert video?**

**The full script:**

We have included plotting the exact solution is this script.

.. code-block:: python

	'''
	Demo for 1D transient diffusion equation on an interval [0,1]
	dc/dt = D*d^2c/dx^2 + f
	f = t*sin(pi*x)
	dc/dx[x=0] = 0
	dc/dx[x=1] = 0
	c[t=0] = sin(pi*x)
	exact solution: c = alpha*gamma*sin(pi*x)
	alpha = exp(-D*pi**2*t)/(D**2 * pi**4)
	gamma = 1 + D**2*pi**4 + exp(D*pi**2*t)*(D*pi**2*t - 1)

	This demo demonstrates how to transient problem
	using the ScalarTransport class
	'''

	import numpy as np
	import matplotlib.pyplot as plt

	# ------------------------------------------------------- #

	import fenics as fe
	from feFlow.physics import TGScalarTransport 
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver

	# Define mesh
	ne = 10
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
	dt = 1e-3
	st = TGScalarTransport(mesh, dt, theta=0.5)
	st.set_element('CG', 1)
	st.set_function_space()

	# Set coefficients on each term
	# here since we are in transient mode, we have to set
	# the function defining the previous and current time step.
	# Since D and u are constants, we repeat the values for both entry
	D = 0.2
	st.set_advection_velocity(0, 0)
	st.set_diffusion_coefficient(D, D)

	# For the reaction term, we have a function that is time dependent
	# we will create two separate functions f0 and fn (currently defined the same way)
	# in the time stepping, we will update f0 and fn with the appopriate t
	f0 = fe.Expression("t*sin(pi*x[0])", degree=1, t=0)
	fn = fe.Expression("t*sin(pi*x[0])", degree=1, t=0)
	st.set_reaction(f0, fn)

	# Set weak form
	st.set_weak_form()

	# Set initial condition
	c0 = fe.interpolate(fe.Expression("sin(pi*x[0])", degree=1), st.V)
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

	# Define exact solution for benchmark
	x = np.linspace(0, 1, ne+1)
	def c_exact(t):
	    alpha = np.exp(-D*np.pi**2*t)/(D**2 * np.pi**4)
	    gamma = 1 + D**2*np.pi**4 + np.exp(D*np.pi**2*t)*(D*np.pi**2*t - 1)
	    return alpha*gamma*np.sin(np.pi*x)

	# Set time steps
	t = 0
	t_end = 0.2
	i = 0
	while t <= t_end:

	    # Update reaction term
	    f0.t = t
	    fn.t = t + dt

	    # Update time
	    t = t + dt

	    # Solve
	    solver.solve()

	    # Update previous solution
	    st.update_sol()

	    # Plot solution against actual solution
	    sol_exact = c_exact(t)
	    fe.plot(st.current_sol, label='Computed solution')
	    plt.plot(x, sol_exact, 'r--', label='Exact solution')
	    plt.legend()
	    plt.title('t = %.4f' % t)
	    plt.ylim([0,1])
	    plt.pause(0.1)
	    plt.cla()

