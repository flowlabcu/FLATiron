from ..info.messages import import_fenics
fe = import_fenics()

from .physics_problem import PhysicsProblem
from .multiphysics_problem import MultiPhysicsProblem
from ..io import *
from .stokes_flow import StokesFlow

class SteadyIncompressibleNavierStokes(StokesFlow):

    def set_initial_guess(self, initial_guess_u=None, initial_guess_p=None):

        # Assign u
        if initial_guess_u is None:
            uini = fe.Constant([0 for i in range(self.dim)])
        else:
            uini = initial_guess_u
        if type(uini) == fe.function.constant.Constant or type(uini) == fe.function.expression.Expression:
            uini = fe.interpolate(uini, self.V.sub(0).collapse())
        fe.assign(self.solution_function().sub(0), uini)

        # Assign p
        if initial_guess_p is None:
            pini = fe.Constant(0)
        else:
            pini = initial_guess_p
        if type(pini) == fe.function.constant.Constant or type(pini) == fe.function.expression.Expression:
            pini = fe.interpolate(pini, self.V.sub(1).collapse())
        fe.assign(self.solution_function().sub(1), pini)

    def set_weak_form(self):

        # Set functions
        V = self.V
        mu = self.external_function('dynamic viscosity')
        rho = self.external_function('density')
        body_force = self.external_function('body force')
        dx = self.dx
        u = self.solution_function('u')
        p = self.solution_function('p')
        w = self.test_function('u')
        q = self.test_function('p')

        # Build weak form
        T = self.stress(u, p, mu)
        F = fe.inner(T, fe.grad(w)) + fe.inner(rho*fe.grad(u)*u, w) + q*fe.div(u)
        self.weak_form = F*self.dx

    def get_residue(self, u, p):
        mu = self.external_function('dynamic viscosity')
        rho = self.external_function('density')
        body_force = self.external_function('body force')
        T = self.stress(u, p, mu)
        return rho*fe.grad(u)*u - fe.div(T) - body_force

    def stress(self, u, p, mu):
        I = fe.Identity(self.dim)
        strain = self.symmetric_strain_rate(u)
        return -p*I + 2*mu*strain

    def symmetric_strain_rate(self, u):
        return 0.5*(fe.grad(u) + fe.grad(u).T)

    def add_stab(self):
        u = self.solution_function('u')
        p = self.solution_function('p')
        w = self.test_function('u')
        q = self.test_function('p')
        rho = self.external_function('density')
        h = self.mesh.cell_diameter()
        r = self.get_residue(u, p)
        (tau_SUPG, tau_PSPG) = self.get_stab_constant(u, h, 1.0, 1.0)
        stab_SUPG = tau_SUPG*fe.inner(fe.grad(w)*u, r)
        stab_PSPG = 1/rho * tau_PSPG * fe.inner(fe.grad(q), r)
        self.add_to_weakform(stab_SUPG, self.dx)
        self.add_to_weakform(stab_PSPG, self.dx)

    def get_stab_constant(self, u, h, alpha, beta):
        return self._T_SUPG(u, h, alpha), self._T_PSPG(u, h, beta)

    def _T_SUPG(self, u, h, alpha):
        mu = self.external_function('dynamic viscosity')
        rho = self.external_function('density')
        u2 = fe.dot(u,u)
        nu = mu/rho
        return alpha * ( (4*u2)/h**2 + 9*(4*nu/h**2)**2 ) ** (-0.5)

    def _T_PSPG(self, u, h, beta):
        mu = self.external_function('dynamic viscosity')
        rho = self.external_function('density')
        u2 = fe.dot(u,u)
        nu = mu/rho
        return beta * ( (4*u2)/h**2 + 9*(4*nu/h**2)**2 ) ** (-0.5)

    def set_dynamic_viscosity(self, mu):
        self.set_external_function('dynamic viscosity', mu)

    def set_density(self, rho):
        self.set_external_function('density', rho)



