from ..info.messages import import_fenics
fe = import_fenics()
from .steady_incompressible_navier_stokes import SteadyIncompressibleNavierStokes
from .incompressible_navier_stokes import IncompressibleNavierStokes


# I take these functions out so that I can call them
# from both the steady and the transient case
# Within the method, the inputs will be kept arbitrary
# so I  can freely change the inputs in these external functions

def _set_ficdom_constants(phys, c1, c2):
    phys.set_external_function('FD_c1', c1)
    phys.set_external_function('FD_c2', c2)

def _set_ficdom_domain(phys, I):
    phys.set_external_function('FD_domain', I)

def _set_ficdom_velocity(phys, u0):
    phys.set_external_function('FD_velocity', u0)

def _fictitious_domain_term(phys, u):
    c1 = phys.external_function('FD_c1')
    c2 = phys.external_function('FD_c2')
    I = phys.external_function('FD_domain')
    u0 = phys.external_function('FD_velocity')
    h = phys.mesh.cell_diameter()
    ficdom_term = I*c1*(1/h)**c2*(u-u0)
    return ficdom_term

class SteadyFicDomNSE(SteadyIncompressibleNavierStokes):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_ficdom_constants(500, 2)
        self.set_ficdom_velocity(fe.Constant([0 for _ in range(self.mesh.dim)]))

    def set_weak_form(self):
        super().set_weak_form()
        u = self.solution_function('u')
        w = self.test_function('u')
        fd_term = self.fictitious_domain_term(u)
        self.weak_form += fe.dot(w, fd_term)*self.dx

    def set_ficdom_constants(self, *args, **kwargs):
        _set_ficdom_constants(self, *args, **kwargs)

    def set_ficdom_domain(self, *args, **kwargs):
        _set_ficdom_domain(self, *args, **kwargs)

    def set_ficdom_velocity(self, *args, **kwargs):
        _set_ficdom_velocity(self, *args, **kwargs)

    def fictitious_domain_term(self, *args, **kwargs):
        return _fictitious_domain_term(self, *args, **kwargs)

class FicDomNSE(IncompressibleNavierStokes):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_ficdom_constants(500, 2)
        self.set_ficdom_velocity(fe.Constant([0 for _ in range(self.mesh.dim)]))

    def set_weak_form(self):
        super().set_weak_form()
        un = self.solution_function('u')
        (u0, p0) = fe.split(self.previous_solution)
        w = self.test_function('u')
        theta = self.external_function('mid point theta')
        fd_term_0 = self.fictitious_domain_term(u0)
        fd_term_n = self.fictitious_domain_term(un)

        fd0 = fe.dot(w, fd_term_0)*self.dx
        fdn = fe.dot(w, fd_term_n)*self.dx
        self.weak_form += (1-theta)*fd0 + theta*fdn

    # def get_residue(self):
    #     """
    #     Here I am computing the residue of the fictitious domain formulation where
    #     the contribution from the fictitious domain term is based on the previous 
    #     timestep velocity only. I am doing this to keep the Krylov solver stable.
    #     """
    #     (u0, p0) = fe.split(self.previous_solution)
    #     r = super().get_residue()
    #     fd_term = self.fictitious_domain_term(u0)
    #     r += fd_term
    #     return r

    def set_ficdom_constants(self, *args, **kwargs):
        _set_ficdom_constants(self, *args, **kwargs)

    def set_ficdom_domain(self, *args, **kwargs):
        _set_ficdom_domain(self, *args, **kwargs)

    def set_ficdom_velocity(self, *args, **kwargs):
        _set_ficdom_velocity(self, *args, **kwargs)

    def fictitious_domain_term(self, *args, **kwargs):
        return _fictitious_domain_term(self, *args, **kwargs)


