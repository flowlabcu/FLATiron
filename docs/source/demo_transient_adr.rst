Transient Advection-Diffusion-Reaction problem
==============================================

This demo demonstrates how to solve a 1D transient Advection-Diffusion-Reaction problem. The full source code can be found in **demo/user_defined/documented/transient_adr_1D/demo_transient_adr_1D.py**

The following problem was taken from Problem 1 in this `paper <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/CE047p0241>`_


Problem definition
--------------------

The strong form

.. math::

	\frac{\partial c}{\partial t} + u \frac{\partial c}{\partial c} = D \frac{\partial^2 c}{\partial x^2}

In this demo, we solve a transport of a Gaussian pulse under a time-varying advection velocity :math:`u(t)` and a fixed diffusivity :math:`D`. The analytical solution of this problem is

.. math::

	c(x,t) = \frac{\sigma_0}{\sigma}exp\left(-\frac{(x-\bar{x})^2}{2\sigma^2}\right) \\

where

.. math::

	\sigma^2 = \sigma_0^2 +2Dt \\

	\bar{x} = x_0 + \int_0^t u(\tau) d\tau \\

and :math:`\sigma_0` and :math:`x_0` are constant initial Gaussian standard deviation and center respectively. From the analytical solution, one can devise the initial condition by simply substituting :math:`t=0` in the above expressions.

In the `paper <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/CE047p0241>`_, the boundary condition is :math:`c(x,t) \rightarrow 0` as :math:`|x| \rightarrow \infty`. To approximate this boundary condition in a finite domain, we will set :math:`c(x,t)=0` at both boundary points, and ensure that the Gaussian pulse is sufficiently far enough from both boundaries.

Implementation
--------------------

We first import relevant libraries and define a mesh using the LineMesh object.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import fenics as fe
    from flatiron_tk.physics import TransientScalarTransport
    from flatiron_tk.mesh import Mesh, LineMesh
    from flatiron_tk.solver import PhysicsSolver

    # Define mesh
    ne = 128
    h = 12800/ne
    mesh = LineMesh(0, 12800, h)

Next, we define several functions for the exact solution of the problem. It is noted that we will use the exact solution to create the initial condition for our numerical setup.

.. code-block:: python

    # Defines x_bar
    def get_x_bar(a_a, a_b, a_t):
        x_bar = (a_a - a_a * np.cos(a_b * a_t)) / a_b
        return x_bar

    # Defines sigma
    def get_sigma(a_sigma_0, a_D, a_t):
        sigma = np.sqrt(a_sigma_0**2 + 2 * a_D * a_t)
        return sigma

    # Defines exact solution
    def get_c_exact(a_x, a_a, a_b, a_t, a_D, a_sigma_0):
        sigma = get_sigma(a_sigma_0, a_D, a_t)
        x_bar = get_x_bar(a_a, a_b, a_t)
        c = (a_sigma_0 / sigma) * np.exp(-1 * (a_x - x_bar)**2 / (2 * sigma**2))
        return c

Next, I define the relevant physics. In this case, it is the ``TransientScalarTransport`` physics. We will integrate this problem in time using :math:`\Delta t = 96` similar to the problem in the benchmark `paper <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/CE047p0241>`_. It is noted that this method integrates the problem using the Crank-Nicolson midpoint algorithm. Please see :doc:`Transient scalar transport physics <transient_scalar_transport_problem>` for the time discretization details.


.. code-block:: python

    # Define problem
    dt = 96
    st = TransientScalarTransport(mesh, dt, tag='c')
    st.set_element('CG', 1)
    st.build_function_space()

Next, we define the advection, diffusion, and reaction terms. It is noted that the ``set_`` functions in ``TransientScalarTransport`` now accepts two variables. These variables represent the *previous* and *current* values of these terms respectively. If the value is constant in time, simply supply the same values in both inputs. The velocity functions ``u0`` and ``un`` are set as FEniCS ``Expression`` objects initially with the same :math:`t` value. We will set the value of :math:`t` during the time integration step.

.. code-block:: python

    # Diffusivity (here set as a constant)
    D = 2
    st.set_diffusivity(D, D)

    # For the velocity term, we have a time-dependent velocity.
    # We will create two separate functions u0  and un and update
    # them with the appropriate t.
    a = 1.5
    b = 2 * np.pi / 9600
    u0 = fe.Expression('a * sin(b*t)', degree=1, a=a, b=b, t=0)
    un = fe.Expression('a * sin(b*t)', degree=1, a=a, b=b, t=0)
    st.set_advection_velocity(u0, un)

    # Similarly, we create f0 and fn for the reaction term (here set to zero)
    st.set_reaction(0, 0)

    # Set weak form
    st.set_weak_form()

Next, we set the initial and boundary conditions and build the solver

.. code-block:: python

    # Set initial condition
    t_0 = 3000
    sigma_0 = 264
    sigma = get_sigma(sigma_0, D, t_0)
    x_bar = get_x_bar(a, b, t_0)
    c0 = fe.interpolate(fe.Expression('s_0/s * exp(-1*pow(x[0]-x_bar,2)/(2*pow(s,2)))',
	                              s_0=sigma_0, s=sigma, x_bar=x_bar, degree=1), st.V)
    st.set_initial_condition(c0)

    bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.)},
	       2:{'type': 'dirichlet', 'value': fe.Constant(0.)}}
    st.set_bcs(bc_dict)

    # Set problem
    solver = PhysicsSolver(st)

At each time step, the time-marching problem is solved by first updating the time-varying advection velocity. The scalar transport problem is then solved, and the solution is updated using the ``update_previous_solution()`` method within the ``TransientScalarTransport`` class. Finally, the solution for the current time step is written using ``st.write(time_stamp=t)`` and visualized with Matplotlib.

.. code-block:: python

    # Begin transient section
    t = t_0
    t_end = 7200
    x = np.linspace(0, 12800, ne, endpoint=True)
    while t <= t_end:

        # Set velocity at current and previous step
        u0.t = t
        un.t = t + dt

        # Solve
        solver.solve()

        # Write output
        st.write(time_stamp=t)

        # Update previous solution
        st.update_previous_solution()

        # Update time
        t += dt

        # Plot computed solution against exact solution
        sol_exact = get_c_exact(x, a, b, t, D, sigma_0)
        fe.plot(st.solution_function(), label='Computed solution')
        plt.plot(x, sol_exact, 'r--', label='Exact solution')
        plt.legend()
        plt.title('t = %.4f' % t)
        plt.ylim([-0.1, 1.1])
        plt.pause(0.01)
        plt.cla()



