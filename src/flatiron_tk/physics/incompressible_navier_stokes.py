from ..info.messages import import_fenics
fe = import_fenics()

from .physics_problem import PhysicsProblem
from .multiphysics_problem import MultiPhysicsProblem
from ..io import *
from .steady_incompressible_navier_stokes import SteadyIncompressibleNavierStokes

class IncompressibleNavierStokes(SteadyIncompressibleNavierStokes):

    def build_function_space(self):
        super().build_function_space()
        self.previous_solution = fe.Function(self.function_space())

    def set_weak_form(self, stab=False):

        mu = self.external_function('dynamic viscosity')
        rho = self.external_function('density')
        dt = self.external_function('dt')
        theta = self.external_function('mid point theta')

        h = self.mesh.cell_diameter()
        w = self.test_function('u')
        q = self.test_function('p')
        un = self.solution_function('u')
        pn = self.solution_function('p')
        (u0, p0) = fe.split(self.previous_solution)
        gw = fe.grad(w)         ; dw = fe.div(w)

        # Main weak formulation
        Tn = self.stress(un, pn, mu)
        F1 = fe.inner( Tn, gw )  + fe.inner( rho*fe.grad(un)*un, w ) - q*fe.div(un)
        T0 = self.stress(u0, pn, mu)
        F0 = fe.inner( T0, gw ) + fe.inner( rho*fe.grad(u0)*u0, w )
        self.weak_form = fe.inner( rho*(un-u0)/dt , w ) + (1-theta)*F0 + theta*F1
        self.weak_form *= self.dx

        # Add stab if option specify
        if stab:
            self.add_stab()

    def add_stab(self):

        rho = self.external_function('density')
        dt = self.external_function('dt')
        theta = self.external_function('mid point theta')

        h = self.mesh.cell_diameter()
        w = self.test_function('u')
        q = self.test_function('p')
        un = self.solution_function('u')
        pn = self.solution_function('p')
        (u0, p0) = fe.split(self.previous_solution)

        # Add stabilization
        r = self.get_residue()
        stab_SUPG = self._T_SUPG(u0, h, 1)*fe.inner(fe.grad(w)*u0, r)
        stab_PSPG = 1/rho*self._T_PSPG(u0, h, 1)*fe.inner(fe.grad(q), r)
        self.add_to_weakform(stab_SUPG, self.dx)
        self.add_to_weakform(stab_PSPG, self.dx)

    def get_residue(self):

        rho = self.external_function('density')
        dt = self.external_function('dt')
        theta = self.external_function('mid point theta')
        un = self.solution_function('u')
        pn = self.solution_function('p')
        (u0, p0) = fe.split(self.previous_solution)
        r0 = super().get_residue(u0, pn)
        rn = super().get_residue(un, pn)
        r = rho*(un-u0)/dt + theta*rn + (1-theta)*r0
        return r

    def _T_SUPG(self, u, h, alpha):
        mu = self.external_function('dynamic viscosity')
        rho = self.external_function('density')
        dt = self.external_function('dt')
        theta = self.external_function('mid point theta')
        tdt = theta*dt
        nu = mu/rho
        u2 = fe.dot(u, u)
        return alpha * ( (1/tdt)**2 + (4*u2)/h**2 + 9*(4*nu/h**2)**2 ) ** (-0.5)

    def _T_PSPG(self, u, h, beta):
        mu = self.external_function('dynamic viscosity')
        rho = self.external_function('density')
        dt = self.external_function('dt')
        theta = self.external_function('mid point theta')
        tdt = theta*dt
        nu = mu/rho
        u2 = fe.dot(u, u)
        return beta * ( (1/tdt)**2 + (4*u2)/h**2 + 9*(4*nu/h**2)**2 ) ** (-0.5)

    def set_initial_conditions(self, u, p):
        fe.assign(self.solution_function('u'), u)
        fe.assign(self.solution_function('p'), p)
        fe.assign(self.mmt.previous_solution, u)
        fe.assign(self.cont.previous_solution, p)

    def update_previous_solution(self):
        self.previous_solution.assign(self.solution_function())

    ####################################
    ## Set physics properties
    ####################################
    def set_time_step_size(self, dt):
        self.set_external_function('dt', dt)

    def set_mid_point_theta(self, new_theta):
        self.set_external_function('mid point theta', new_theta)
