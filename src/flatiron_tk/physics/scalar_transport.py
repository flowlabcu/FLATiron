'''
Defines a scalar transport problem
'''
from ..info.messages import import_fenics
fe = import_fenics()
from .physics_problem import PhysicsProblem

class ScalarTransport(PhysicsProblem):

    # -----------------------------------------
    # Getters for external functions
    # -----------------------------------------

    def set_advection_velocity(self, u):
        self.set_external_function('advection velocity', u)

    def set_diffusivity(self, D):
        self.set_external_function('diffusivity', D)

    def set_reaction(self, R):
        self.set_external_function('reaction', R)

    def get_advection_velocity(self):
        return self.external_function('advection velocity')

    def get_diffusivity(self):
        return self.external_function('diffusivity')

    def get_reaction(self):
        return self.external_function('reaction')

    # -----------------------------------------
    # Weak formulation
    # -----------------------------------------
    def set_weak_form(self, stab=False):

        '''
        Set weak formulation as:
        A - D - R = 0
        where

        A, D, R are the weak form terms relating to the advection, diffusion, and reaction term in the final weak form
        '''

        # If a problem is linear, use trial function for straight forward
        # Bilinear/linear assemble
        # otherwise use the solution funciton
        c = self.solution_function()
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()

        # Grab each term in ADR equation
        a_form = self.advection(u, c)
        d_form = self.diffusion(D, c)
        r_form = self.reaction(r, c)

        # Set weakform
        self.weak_form = a_form - d_form - r_form

        # Add stabilization
        if stab:
            self.add_stab()

    def advection(self, u, c, domain=None):
        w = self.test_function()
        if domain is None: domain = self.dx
        if self.dim == 1:
            form = w*u*fe.grad(c)[0]*domain
        else:
            form = w*fe.dot(u, fe.grad(c))*domain
        return form

    def diffusion(self, D, c, domain=None):
        w = self.test_function()
        if domain is None: domain = self.dx
        form = -D*fe.inner(fe.grad(w), fe.grad(c))*domain
        return form

    def reaction(self, r, c, domain=None):
        w = self.test_function()
        if domain is None: domain = self.dx
        form = w*r*domain
        return form

    def flux(self, h):
        w = self.test_function()
        n = self.mesh.facet_normal()
        if self.dim == 1:
            form = -w*h*fe.dot(fe.Constant((1,)), n)
        else:
            form = -w*fe.dot(h, n)
        return form

    # -----------------------------------------
    # Stabilization
    # -----------------------------------------
    def add_stab(self, tau_type='shakib'):

        # Get functions
        w = self.test_function()
        h = self.mesh.cell_diameter()
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()

        # Get SU term
        if self.dim == 1:
            p = u*fe.grad(w)[0]
        else:
            p = fe.dot(u, fe.grad(w))

        # Get stabilization constant
        tau = self.get_stab_constant(tau_type)

        # Add stabilization term
        residue = self.get_residue()
        self.add_to_weakform(p*tau*residue, self.dx)

    def get_stab_constant(self, tau_type):

        # Make sure tau type is available
        avail_tau = ['shakib', 'su', 'codina']
        try:
            assert(tau_type in avail_tau)
        except:
            raise ValueError("tau_type %s is not available. Available tau_type are %s" % (tau_type, avail_tau))

        # Grab functions
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()
        h = self.mesh.cell_diameter()

        # Get |u|
        if self.dim == 1:
            u_norm = fe.sqrt(u*u)
        else:
            u_norm = fe.sqrt(fe.dot(u, u))
        Pe = u_norm*h/2/D # Cell Peclet number

        # Define tau
        if tau_type == 'shakib':
            tau = ( (2*u_norm/h)**2 + 9*(4*D/h**2)**2 + r**2 )**(-0.5)
        elif tau_type == 'su':
            coth = lambda a: (fe.exp(a) + fe.exp(-a))/(fe.exp(a) - fe.exp(-a)) # coth(a) in exponential form
            tau = h/2/u_norm*(coth(Pe) - 1/Pe)
        elif tau_type == 'codina':
            tau = h/2/u_norm*(1 + 1/Pe + h*r/2/u_norm)**(-1)
        return tau

    # -----------------------------------------
    # Residue
    # -----------------------------------------
    def get_residue(self):

        # Grab functions
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()
        c = self.solution_function()

        # Advection residue
        if self.dim == 1:
            a_residue = u*fe.grad(c)[0]
        else:
            a_residue = fe.dot(u, fe.grad(c))

        # Diffusion residue
        d_residue = -D*fe.div(fe.grad(c))

        # Reaction residue
        r_residue = r

        # Return total residue
        return a_residue + d_residue - r_residue



