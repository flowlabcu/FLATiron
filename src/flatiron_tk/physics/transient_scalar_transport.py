from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()

from flatiron_tk.physics import SteadyScalarTransport


class TransientScalarTransport(SteadyScalarTransport):
    def __init__(self, mesh, dt=0.01, theta=0.5, *args, **kwargs):
        """
        Transient scalar transport problem. Supers SteadyScalarTransport.

        Parameters
        -------------
        mesh : flatiron_tk.mesh
            The mesh on which to solve the problem.
        dt : float
            The time step size.
        theta : float, optional
            The theta parameter for the implicit-explicit scheme. Default is 0.5.
        """
        super().__init__(mesh, *args, **kwargs)
        
        self.set_external_function('dt', dt)
        self.set_external_function('midpoint_theta', theta)
        self.set_tag('c')

    def build_function_space(self):
        """
        Build the function space for the transient scalar transport problem.
        """
        super().build_function_space()
        self.previous_solution = dolfinx.fem.Function(self.get_function_space())

    def set_advection_velocity(self, u0, un):
        """
        Set the advection velocity for the transient scalar transport problem.
        
        Parameters:
        -----------
            u0: The advection velocity at the previous time step.
            un: The advection velocity at the current time step.
        """
        self.set_external_function('previous_advection_velocity', u0)
        self.set_external_function('advection_velocity', un)

    def set_diffusivity(self, D0, Dn):
        """
        Set the diffusivity for the transient scalar transport problem.
        
        Parameters:
        -----------
            D0: The diffusivity at the previous time step.
            Dn: The diffusivity at the current time step.
        """
        self.set_external_function('previous_diffusivity', D0)
        self.set_external_function('diffusivity', Dn)

    def set_reaction(self, R0, Rn):
        """
        Set the reaction term for the transient scalar transport problem.
        
        Parameters:
        -----------
            R0: The reaction term at the previous time step.
            Rn: The reaction term at the current time step.
        """
        self.set_external_function('previous_reaction', R0)
        self.set_external_function('reaction', Rn)

    def set_time_step_size(self, dt):
        """
        Set the time step size for the transient scalar transport problem.
        
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
        Set weak formulation as:
        dc/dt + mid(advection) - mid(diffusion) - mid(reaction) == 0

        for a mid point type method
        where mid(f) = theta*f1 + (1-theta)*f0

        Parameters
        -----------
            stab: A boolean indicating whether to include stabilization terms in the weak form.
        """
        dt = self.external_function('dt')
        theta = self.external_function('midpoint_theta')
        
        w = self.get_test_function()
        c0 = self.previous_solution
        c = self.get_solution_function()
        D0 = self.external_function('previous_diffusivity')
        Dn = self.external_function('diffusivity')
        u0 = self.external_function('previous_advection_velocity')
        un = self.external_function('advection_velocity')
        r0 = self.external_function('previous_reaction')
        rn = self.external_function('reaction')

        # Transient weakform 
        dcdt = w * (c - c0) / dt * self.dx

        # Previous terms 
        a_form_0 = self._advective_form(c0, u0)
        d_form_0 = self._diffusive_form(c0, D0)
        r_form_0 = self._reactive_form(r0)

        # Current terms
        a_form_n = self._advective_form(c, un)
        d_form_n = self._diffusive_form(c, Dn)
        r_form_n = self._reactive_form(rn)

        # Set forms 
        form_0 = a_form_0 + d_form_0 - r_form_0
        form_n = a_form_n + d_form_n - r_form_n

        # Define weak form by theta-galerkin method
        self.weak_form = dcdt + (1 - theta) * form_0 + theta * form_n

        if stab:
            self.add_stab()

    def add_stab(self):
        """
        Add stabilization term to the weak form.
        """

        w = self.get_test_function()
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        
        if self.mesh.get_gdim() == 1:
            P = u * ufl.grad(w)[0]
        else:
            P = ufl.dot(u, ufl.grad(w))
        residual = self.get_residual()
        tau = self.stabilization_constant()
        self.add_to_weak_form(P * tau * residual, self.dx)

        return 

    def stabilization_constant(self):
        """
        tau from Donea's book Remark 5.8
        """

        dt = self.external_function('dt')
        w = self.get_test_function()
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()
        h = self.mesh.get_cell_diameter()
        theta = self.external_function('midpoint_theta')

        # define the normal of u 
        u_norm = ufl.sqrt(u*u) if self.mesh.get_gdim() == 1 else ufl.sqrt(ufl.dot(u, u))

        # define tau 
        tau = ( (1/(theta*dt))**2 + (2*u_norm/h)**2 + 9*(4*D/h**2)**2 + r**2 ) **(-0.5)

        return tau
    
    def get_residual(self):
        """
        Compute the residual for the transient scalar transport problem.
        
        Returns:
        -------------
            The residual expression for the transient scalar transport equations.
        """
        dt = self.external_function('dt')
        theta = self.external_function('midpoint_theta')

        c0 = self.previous_solution
        D0 = self.external_function('previous_diffusivity')
        u0 = self.external_function('previous_advection_velocity')
        r0 = self.external_function('previous_reaction')

        c = self.get_solution_function()
        Dn = self.external_function('diffusivity')
        un = self.external_function('advection_velocity')
        rn = self.external_function('reaction')

        # advective residual
        if self.mesh.get_gdim() == 1:
            a_residual = theta*un*ufl.grad(c)[0] + (1-theta)*u0*ufl.grad(c0)[0]
        else:
            a_residual = theta*ufl.dot(un, ufl.grad(c)) + (1-theta)*ufl.dot(u0, ufl.grad(c0))

        # diffusive residual
        d_residual = -theta*Dn*ufl.div(ufl.grad(c)) + (1-theta)*D0*ufl.div(ufl.grad(c0))

        # reactive residual
        r_residual = theta*rn + (1-theta)*r0

        dcdt = (c - c0) / dt

        return dcdt + a_residual + d_residual - r_residual
    
    def set_initial_condition(self, u0):
        """
        Set the initial condition for the problem.

        Parameters
        -------------
        u0 : dolfinx.fem.Function
            The initial condition function.
        """
        self.previous_solution.interpolate(u0)
        self.previous_solution.x.array[:] = u0.x.array[:]


    def update_previous_solution(self):
        """
        Update the previous solution with the current solution.
        """
        self.previous_solution.x.array[:] = self.get_solution_function().x.array[:]