from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()

from flatiron_tk.physics import SteadyNavierStokes

class TransientNavierStokes(SteadyNavierStokes):
    """
    Transient scalar transport problem. Supers SteadyNavierStokes.

    Parameters
    -------------
    mesh : flatiron_tk.mesh
        The mesh on which to solve the problem.
    dt : float
        The time step size.
    theta : float, optional
        The theta parameter for the implicit-explicit scheme. Default is 0.5.
    """
    def __init__(self, mesh, dt=0.01, theta=0.5, *args, **kwargs):
        super().__init__(mesh, *args, **kwargs)
        
        self.set_external_function('dt', dt)
        self.set_external_function('midpoint_theta', theta)

    def build_function_space(self):
        """
        Build the function space for the transient Navier-Stokes problem.
        """
        super().build_function_space()
        self.previous_solution = dolfinx.fem.Function(self.get_function_space())

    def set_time_step_size(self, dt):
        """
        Set the time step size for the transient Navier-Stokes problem.
        
        Parameters:
        -----------
            dt: The time step size.
        """
        self.set_external_function('dt', dt)

    def set_midpoint_theta(self, theta):
        """
        Set the theta parameter for the implicit-explicit scheme.
        
        Parameters:
        -----------
            theta: The theta parameter.
        """
        self.set_external_function('midpoint_theta', theta)
    
    def set_weak_form(self, stab=False):
        """
        Set the weak form for the transient Navier-Stokes problem.
        
        Parameters:
        -----------
            stab: A boolean indicating whether to include stabilization terms in the weak form.
        """
        mu = self.external_function('dynamic_viscosity')
        rho = self.external_function('density')
        body_force = self.external_function('body_force')
        dx = self.dx
        u = self.get_solution_function('u')
        p = self.get_solution_function('p')
        
        
        theta = self.external_function('midpoint_theta')
        dt = self.external_function('dt')
        u0, p0 = ufl.split(self.previous_solution)
        
        w = self.get_test_function('u')
        q = self.get_test_function('p')

        T0_1 = rho * ufl.inner(w, ufl.grad(u)*u) * dx 
        T1_1 = mu * ufl.inner(ufl.grad(w), ufl.grad(u)) * dx 
        T2_1 = p * ufl.div(w) * dx
        T3_1 = q * ufl.div(u) * dx
        T4_1 = rho * ufl.dot(w, body_force) * dx
        form_1 = T0_1 + T1_1 - T2_1 - T3_1 - T4_1

        T0_0 = rho * ufl.inner(w, ufl.grad(u0)*u0) * dx
        T1_0 = mu * ufl.inner(ufl.grad(w), ufl.grad(u0)) * dx
        T2_0 = p * ufl.div(w) * dx
        T3_0 = q * ufl.div(u0) * dx
        T4_0 = rho * ufl.dot(w, body_force) * dx
        form_0 = T0_0 + T1_0 - T2_0 - T3_0 - T4_0

        self.weak_form = ufl.inner(rho * (u - u0) / dt, w) * dx + (1 - theta) * form_0 + theta * form_1

        if stab:
            self.add_stab()

    def add_stab(self):
        """
        Add stabilization terms to the weak form for the steady Navier-Stokes problem.
        This method computes the SUPG and PSPG stabilization terms and adds them to the weak form.
        """
        u = self.get_solution_function('u')
        p = self.get_solution_function('p')
        w = self.get_test_function('u')
        q = self.get_test_function('p')
        rho = self.external_function('density')
        h = self.mesh.get_cell_diameter()

        u0, _ = ufl.split(self.previous_solution)

        r = self.get_residual()
        stab_SUPG = self._T_SUPG(u0, h, 1.0) * ufl.inner(ufl.grad(w)*u, r)
        stab_PSPG = -1 / rho * self._T_PSPG(u0, h, 1.0) * ufl.inner(ufl.grad(q), r)

        self.add_to_weak_form(stab_SUPG, self.dx)
        self.add_to_weak_form(stab_PSPG, self.dx)

    def get_residual(self):
        """
        Compute the residual for the transient Navier-Stokes problem.
        
        Returns:
        -------------
            The residual expression for the transient Navier-Stokes equations.
        """
        rho = self.external_function('density')
        u = self.get_solution_function('u')
        p = self.get_solution_function('p')
        theta = self.external_function('midpoint_theta')
        dt = self.external_function('dt')
        u0, _ = ufl.split(self.previous_solution)

        r0 = super().get_residual(u0, p)
        rn = super().get_residual(u, p)

        return rho * ((u - u0) / dt) + (1 - theta) * r0 + theta * rn


    def _T_SUPG(self, u, h, alpha):
        """
        Compute the SUPG stabilization term for the steady Navier-Stokes problem.
        
        Parameters:
        -------------
            u: The velocity field.
            h: The cell diameter.
            alpha: The SUPG stabilization parameter.
        
        Returns:
        -------------
            The SUPG stabilization term.
        """
        mu = self.external_function('dynamic_viscosity')
        rho = self.external_function('density')
        theta = self.external_function('midpoint_theta')
        dt = self.external_function('dt')

        u2 = ufl.dot(u,u)
        nu = mu/rho

        return alpha * (1 / (theta * dt)**2 + 4 * u2 / h**2 + 9 * (4 * nu / h**2)**2) ** (-0.5)
                          

    def _T_PSPG(self, u, h, beta):
        """
        Compute the PSPG stabilization term for the steady Navier-Stokes problem.
        
        Parameters:
        -------------
            u: The velocity field.
            h: The cell diameter.
            beta: The PSPG stabilization parameter.
        
        Returns:
        -------------
            The PSPG stabilization term.
        """
        mu = self.external_function('dynamic_viscosity')
        rho = self.external_function('density')
        theta = self.external_function('midpoint_theta')
        dt = self.external_function('dt')

        u2 = ufl.dot(u,u)
        nu = mu/rho

        return beta * (1 / (theta * dt)**2 + 4 * u2 / h**2 + 9 * (4 * nu / h**2)**2) ** (-0.5)
    
    def set_initial_conditions(self, u_init, p_init):
        """
        Set the initial conditions for the transient Navier-Stokes problem.

        Parameters:
        -------------
            u_init: dolfinx.fem.Function or dolfinx.fem.Expression
                Initial velocity field
            p_init: dolfinx.fem.Function or dolfinx.fem.Expression
                Initial pressure field
        """
        # Get subfunctions for current and previous solutions
        u_curr, p_curr = self.solution.sub(0), self.solution.sub(1)
        u_prev, p_prev = self.previous_solution.sub(0), self.previous_solution.sub(1)

        # Helper function to assign or interpolate
        def assign_or_interpolate(subfunc, init):
            if isinstance(init, dolfinx.fem.Function):
                subfunc.vector.set(init.vector.array)
            else:
                subfunc.interpolate(init)
            subfunc.vector.scatter_forward()

        # Set velocity and pressure for both current and previous solutions
        for subfunc, init in [(u_curr, u_init), (p_curr, p_init),
                            (u_prev, u_init), (p_prev, p_init)]:
            assign_or_interpolate(subfunc, init)

    def update_previous_solution(self):
        """
        Update the previous solution with the current solution.
        This method copies the current solution to the previous solution for the next time step.
        """
        self.previous_solution.x.array[:] = self.solution.x.array[:]



        



    