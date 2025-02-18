'''
Base class for transient problems for scalar transports
'''


from ..info.messages import import_fenics
fe = import_fenics()

from .scalar_transport import ScalarTransport

class TransientScalarTransport(ScalarTransport):

    # -----------------------------------------
    # Init
    # -----------------------------------------
    def __init__(self, mesh, dt, theta=0.5, *args, **kwargs):
        super().__init__(mesh, *args, **kwargs)
        self.set_external_function('dt', dt)
        self.set_external_function('midpoint theta', theta)

    def build_function_space(self):
        super().build_function_space()
        self.previous_solution = fe.Function(self.function_space())

    # -----------------------------------------
    # Getters for external functions. Overloaded
    # to include the previous value
    # -----------------------------------------
    def set_advection_velocity(self, u0, un):
        self.set_external_function('previous advection velocity', u0)
        self.set_external_function('advection velocity', un)

    def set_diffusivity(self, D0, Dn):
        self.set_external_function('previous diffusivity', D0)
        self.set_external_function('diffusivity', Dn)

    def set_reaction(self, R0, Rn):
        self.set_external_function('previous reaction', R0)
        self.set_external_function('reaction', Rn)

    # -----------------------------------------
    # Weak formulation
    # -----------------------------------------
    def set_weak_form(self, stab=False):

        '''
        Set weak formulation as:
        dc/dt + mid(advection) - mid(diffusion) - mid(reaction) == 0

        for a mid point type method
        where mid(f) = theta*f1 + (1-theta)*f0
        '''
        w = self.test_function()

        # Midpoint integration parameters
        dt = self.external_function('dt')
        theta = self.external_function('midpoint theta')

        # Get previous state functions
        c0 = self.previous_solution
        D0 = self.external_function('previous diffusivity')
        u0 = self.external_function('previous advection velocity')
        r0 = self.external_function('previous reaction')

        # Get current state functions
        c = self.solution_function()
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()

        # Create the transient weakform
        dcdt = w*(c-c0)/dt*self.dx

        # Previous time step term
        a_form_0 = self.advection(u0, c0)
        d_form_0 = self.diffusion(D0, c0)
        r_form_0 = self.reaction(r0, c0)

        # Current time step term
        a_form_n = self.advection(u, c)
        d_form_n = self.diffusion(D, c)
        r_form_n = self.reaction(r, c)

        # Set previous and current form
        form_0 = a_form_0 - d_form_0 - r_form_0
        form_n = a_form_n - d_form_n - r_form_n

        # Define weak form via midpoint method
        self.weak_form = dcdt + (1-theta)*form_0 + theta*form_n

        if stab:
            self.add_stab()

    # -----------------------------------------
    # Stabilization
    # -----------------------------------------
    def add_stab(self):
        w = self.test_function()
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        if self.dim == 1:
            P = u*fe.grad(w)[0]
        else:
            P = fe.dot(u, fe.grad(w))

        residue = self.get_residue()
        tau = self.stab_constant()
        self.add_to_weakform(P*tau*residue, self.dx)

    def stab_constant(self):
        '''
        tau from Donea's book Remark 5.8
        '''

        # Grab functions
        dt = self.external_function('dt')
        w = self.test_function()
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()
        h = self.mesh.cell_diameter()
        theta = self.external_function('midpoint theta')
        # Define |u|
        u_norm = fe.sqrt(u*u) if self.dim == 1 else fe.sqrt(fe.dot(u, u))

        # Define tau
        tau = ( (1/(theta*dt))**2 + (2*u_norm/h)**2 + 9*(4*D/h**2)**2 + r**2 ) **(-0.5)
        return tau

    def edge_stab(self, gamma, c):
        theta = self.external_function('midpoint theta')
        h_f = fe.FacetArea(self.mesh.fenics_mesh())
        c0 = self.previous_solution
        J0 = super().edge_stab(gamma, c0)
        Jn = super().edge_stab(gamma, c)
        return (theta)*J0 + (1-theta)*Jn

    def get_residue(self):

        # Midpoint integration parameters
        dt = self.external_function('dt')
        theta = self.external_function('midpoint theta')

        # Get previous state functions
        c0 = self.previous_solution
        D0 = self.external_function('previous diffusivity')
        u0 = self.external_function('previous advection velocity')
        r0 = self.external_function('previous reaction')

        # Get current state functions
        c = self.solution_function()
        D = self.get_diffusivity()
        u = self.get_advection_velocity()
        r = self.get_reaction()

        # Advection residue
        if self.dim == 1:
            a_residue = theta*u*fe.grad(c)[0] + (1-theta)*u0*fe.grad(c0)[0]
        else:
            a_residue = theta*fe.dot(u, fe.grad(c)) + (1-theta)*fe.dot(u0, fe.grad(c0))

        # Diffusion residue
        d_residue = -theta*D*fe.div(fe.grad(c)) + (1-theta)*D0*fe.div(fe.grad(c0))

        # Reaction residue
        r_residue = theta*r + (1-theta)*r0

        # Time dependent term
        dcdt = (c-c0)/dt

        return dcdt + a_residue + d_residue - r_residue

    # -----------------------------------------
    # Initial conditions
    # -----------------------------------------
    def set_initial_condition(self, f):
        fe.assign(self.previous_solution, f)
        fe.assign(self.solution_function(), f)

    # -----------------------------------------
    # Solution update
    # -----------------------------------------
    def update_previous_solution(self):
        self.previous_solution.assign(self.solution_function())

    ####################################
    ## Set physics properties
    ####################################
    def set_time_step_size(self, dt):
        self.set_external_function('dt', dt)

    def set_mid_point_theta(self, new_theta):
        self.set_external_function('mid point theta', new_theta)





