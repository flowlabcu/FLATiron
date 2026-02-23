import ufl 

from flatiron_tk.physics import PhysicsProblem

class SteadyScalarTransport(PhysicsProblem):
    """
    A class to represent the steady scalar transport equations.
    This class extends the PhysicsProblem class to include advection, diffusion, and reaction terms.
    """

    def build_function_space(self, *args, **kwargs):
        """
        Build the function space for the steady scalar transport problem.
        
        Parameters
        ----------
            *args: Arguments to be passed to the parent class method.
            **kwargs: Keyword arguments to be passed to the parent class method.
        """
        # Call the parent class method
        super().build_function_space(*args, **kwargs)
        # Add a name to the soltuion function 
        self.solution.name = self.tag
        
    def set_advection_velocity(self, u):
        """
        Set the advection velocity for the steady scalar transport problem.
        
        Parameters
        -------------
            u: The advection velocity function.
        """
        self.set_external_function('advection_velocity', u)
    
    def set_diffusivity(self, D):
        """
        Set the diffusivity for the steady scalar transport problem.
        
        Parameters
        -------------
            D: The diffusivity function.
        """
        self.set_external_function('diffusivity', D)
    
    def set_reaction(self, R):
        """
        Set the reaction for the steady scalar transport problem.
        
        Parameters
        -------------
            R: The reaction function.
        """
        self.set_external_function('reaction', R)

    def get_advection_velocity(self):
        """
        Get the advection velocity for the steady scalar transport problem.
        
        Returns:
        -------------
            The advection velocity function.
        """
        return self.external_function('advection_velocity')
    
    def get_diffusivity(self):
        """
        Get the diffusivity for the steady scalar transport problem.
        
        Returns:
        -------------
            The diffusivity function.
        """
        return self.external_function('diffusivity')
    
    def get_reaction(self):
        """
        Get the reaction for the steady scalar transport problem.
        
        Returns:
        -------------
            The reaction function.
        """
        return self.external_function('reaction')
    
    # ---- Weak Form ---- #
    def set_weak_form(self, stab=False):
        """
        Set the weak form for the steady scalar transport problem.
        
        Parameters
        -------------
            stab: Whether to include stabilization terms.   
        
        """
        c = self.get_solution_function()
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()

        a_form = self._advective_form(c, u)
        d_form = self._diffusive_form(c, D)
        r_form = self._reactive_form(r)

        self.weak_form = a_form + d_form - r_form

    def _advective_form(self, c, u, domain=None):
        """ 
        Define the advective transport term of the weak form.
        
        Parameters
        -------------
            c: The solution function.
            u: The advection velocity function.
            domain: The domain over which to integrate.
        
        Returns:
        -------------
            form: The advective transport term of the weak form.
        """

        w = self.get_test_function()
        if domain is None:
            domain = self.dx
        
        if self.mesh.get_gdim() == 1:
            form = w * u * ufl.grad(c)[0] * domain
        
        else:
            form = w * ufl.dot(u, ufl.grad(c)) * domain

        return form
    
    def _diffusive_form(self, c, D, domain=None):
        """ 
        Define the diffusive transport term of the weak form.
        
        Parameters
        -------------
            c: The solution function.
            D: The diffusivity function.
            domain: The domain over which to integrate.
        
        Returns:
        -------------
            form: The diffusive transport term of the weak form.
        """
        
        w = self.get_test_function()
        if domain is None:
            domain = self.dx
        
        form = D * ufl.inner(ufl.grad(w), ufl.grad(c)) * domain

        return form
    
    def _reactive_form(self, r, domain=None):
        """
        Define the reactive transport term of the weak form.
        
        Parameters
        -------------
            c: The solution function.
            r: The reaction function.
            domain: The domain over which to integrate.
        
        Returns:
        -------------
            form: The reactive transport term of the weak form.
        """
        w = self.get_test_function()
        if domain is None:
            domain = self.dx
        
        form = w * r * domain

        return form

    def flux(self, q):
        """
        Computes the flux of the problem.
        
        Parameters
        -------------
            q: The flux vector 

        Advection_Matrix + Diffusion_Matrix - Reaction_Matrix - Flux_Term = 0 => we subtract the form 
        """
        w = self.get_test_function()
        n = self.mesh.get_facet_normal()
        if self.mesh.get_gdim() == 1:
            form = -w * q * n[0]
        else:
            form = -w * ufl.dot(q, n)

        
        return form
    
    def add_stab(self, tau_type='shakib'):
        """
        Adds the stabilization term to the weak form.
        
        Parameters
        -------------
            tau_type: The type of stabilization constant to use. Options are 'shakib','su', or 'codina'.
        """
        w = self.get_test_function()
        h = self.mesh.get_cell_diameter()
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()

        # --- SU term --- #
        if self.mesh.get_gdim() == 1:
            P = u * ufl.grad(w)[0]
        else:
            P = ufl.dot(u, ufl.grad(w))

        # --- Stabilization Constant --- #
        tau = self.stabilization_constant(tau_type)
        residual = self.get_residual()
        
        self.add_to_weak_form(P * tau * residual * self.dx)

    def stabilization_constant(self, tau_type):
        """
        Computes the stabilization constant.
        
        Parameters
        -------------
            tau_type: The type of stabilization constant to use. Options are 'shakib','su', or 'codina'.
        
        Returns:
        -------------
            tau: The stabilization constant.
        """
        avail_tau = ['shakib', 'su', 'codina']
        try:
            assert(tau_type in avail_tau)
        except:
            raise ValueError("tau_type %s is not available. Available tau_type are %s" % (tau_type, avail_tau))
        
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()
        h = self.mesh.get_cell_diameter()

        if self.mesh.get_gdim() == 1:
            u_norm = ufl.sqrt(u*u)
        else:
            u_norm = ufl.sqrt(ufl.dot(u, u))

        # --- Cell Peclet Number --- #
        Pe = u_norm * 2 * h / D

        if tau_type == 'shakib':
            tau = ((2 * u_norm / h)**2 + 9*(4 * D / h**2)**2 + r**2)**(-0.5)
        elif tau_type == 'su':
            coth = lambda a: (ufl.exp(a) + ufl.exp(-a)) / (ufl.exp(a) - ufl.exp(-a)) # exponential form
            tau = h / (2 * u_norm) * (coth(Pe) - 1/Pe)
        elif tau_type == 'codina':
            tau = h / (2 * u_norm) * (1 + 1/Pe + h * r / 2 / u_norm)**(-1)
        
        return tau
    
    def add_edge_stab(self, gamma):
        """
        Adds edge stabilization to the weak form.
        
        Parameters
        -------------
            gamma: The edge stabilization parameter.
        """
        c = self.get_solution_function()
        self.add_to_weak_form(self.get_edge_stab(gamma, c))
        
    def get_edge_stab(self, gamma, c):
        """
        Computes the edge stabilization term.
        
        Parameters
        -------------
            gamma: The edge stabilization parameter.
            c: The solution function.
        
        Returns:
        -------------
            J: The edge stabilization term."""

        # NOTE: ufl.FacetArea may not work for higher order meshes (non-teterahedral)
        h_f = ufl.FacetArea(self.mesh.msh)
        w = self.get_test_function()
        n = self.mesh.get_facet_normal()

        if self.mesh.get_tdim() == 1:
            coef = 0.5 * gamma # facet is a point, so we don't have edge length
        else:
            coef = 0.5 * gamma * h_f("+")**self.mesh.get_tdim() # FacetArea("+") is the positive side of each facet
        J = coef * ufl.jump(ufl.inner(ufl.grad(w), n)) * ufl.jump(ufl.inner(ufl.grad(c), n)) * self.dS

        return J
    
    def get_residual(self):
        """
        Computes the residual of the problem.
        
        Returns:
        -------------
            residual: The residual of the problem as a fenicsx form
        """
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()
        c = self.get_solution_function()

        if self.mesh.get_gdim() == 1:
            advection_residual = u * ufl.grad(c)[0]
        else:
            advection_residual = ufl.dot(u, ufl.grad(c))

        diffusion_residual = ufl.div(D * ufl.grad(c))

        reaction_residual = r

        return advection_residual - diffusion_residual - reaction_residual