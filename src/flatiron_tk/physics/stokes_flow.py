
from ..info.messages import import_fenics
fe = import_fenics()

from .physics_problem import PhysicsProblem
from .multiphysics_problem import MultiPhysicsProblem
from ..io import *

class Momentum(PhysicsProblem):
    """"""

    def set_weak_form(self):
        """"""

    def set_element(self, element_family, element_degree):
        self.element = fe.VectorElement(element_family, self.mesh.fenics_mesh().ufl_cell(), element_degree)

    def flux(self):
        """"""

    def get_residue(self):
        """"""

    def get_zero_constant(self):
        """Overload if this is a vector function"""
        return fe.Constant([0. for i in range(self.dim)])

class Continuity(PhysicsProblem):
    """"""

    def set_weak_form(self):
        """"""

    def flux(self):
        """"""

    def get_residue(self):
        """"""


class _OnPoint():
    def __init__(self, point_location, eps):
        self.x = point_location
        self.eps = eps
    def eval(self, x, on_boundary):
        on_point = False
        for i in range(len(self.x)):
            on_point = on_point and fe.near(self.x[i], x[i], self.eps)
        return on_point


class StokesFlow(MultiPhysicsProblem):

    def __init__(self, mesh):
        self.mmt = Momentum(mesh, 'u')
        self.cont = Continuity(mesh, 'p')
        super().__init__(self.mmt, self.cont)

        # Sensible default values
        self.set_external_function('body force', fe.Constant([0 for i in range(self.dim)]))

    def set_element(self, u_family, u_degree, p_family, p_degree):
        self.mmt.set_element(u_family, u_degree)
        self.cont.set_element(p_family, p_degree)
        super().set_element()

    def build_function_space(self):
        super().build_function_space()

    def set_weak_form(self):
        nu = self.external_function('kinematic viscosity')
        body_force = self.external_function('body force')
        dx = self.dx
        u = self.solution_function('u')
        p = self.solution_function('p')
        w = self.test_function('u')
        q = self.test_function('p')

        # Build weak form
        # self.weak_form = a + b + bT - fb
        bT = -q * fe.div(u) * dx
        self.weak_form = self._get_stokes_mmt(u, p, body_force, nu) + bT

    def _get_stokes_mmt(self, u, p, body_force, nu):
        dx = self.dx
        V = self.function_space()
        w = self.test_function('u')
        a = nu * fe.inner(fe.grad(w), fe.grad(u)) * dx
        b = -fe.div(w) * p * dx
        fb = fe.inner(w, body_force) * dx
        return a + b - fb

    def get_stab_constant(self):
        h = self.mesh.cell_diameter()
        nu = self.external_function('kinematic viscosity')
        tau = 1./3.*h**2/4/nu
        return tau

    def add_stab(self):
        (u, p) = fe.split(self.solution_function())
        q = self.test_function('p')
        b = self.external_function('body force')
        P = fe.grad(q) 
        R = fe.grad(p) 
        tau = self.get_stab_constant()
        self.add_to_weakform(fe.dot(P, tau*R), self.dx)

    def flux(self, h, physics_tag):
        w = self.test_function('w')
        n = self.mesh.facet_normal()
        I = fe.Identity(self.dim)
        flux_form = fe.dot(w, -fe.dot(h, n))
        return flux_form

    ####################################
    ## Set physics properties
    ####################################
    def set_kinematic_viscosity(self, nu):
        self.set_external_function('kinematic viscosity', nu)

    def set_body_force(self, b):
        self.set_external_function('body force', b)

    def set_bcs(self, multiphysics_bcs_dict):
        '''
        Here I am overloading the set_bcs from the MultiPhysics because
        the pressure boundary conditions has to be handled separately
        '''

        u_bcs = multiphysics_bcs_dict['u']
        p_bcs = multiphysics_bcs_dict['p']

        # Run MultiPhysics set bcs only on the velocity conditions
        super().set_bcs({'u':u_bcs})

        # Handle pressure boundary conditions
        boundary = self.mesh.boundary
        n = self.mesh.facet_normal()
        w = self.test_function('u')
        pV = self.function_space('p')
        for boundary_id in p_bcs:

            bc_data = p_bcs[boundary_id]
            bc_type = bc_data['type']
            p_val = fe.Constant(0.) if bc_data['value'] == 'zero' else bc_data['value']

            if bc_type == 'dirichlet':

                if isinstance(boundary_id, int):
                    self.dirichlet_bcs.append(fe.DirichletBC(pV, p_val, boundary, boundary_id))

                elif isinstance(boundary_id, str):

                    if not boundary_id.startswith('point'):
                        continue

                    point_location = bc_data['x']
                    # Check if the user provides a tolerance for point search
                    # otherwise use machine precision
                    eps = bc_data['eps'] if 'eps' in bc_data else 3e-16
                    def on_point(x, on_boundary):
                        is_on_point = True
                        for i in range(len(point_location)):
                            is_on_point = is_on_point and fe.near(x[i], point_location[i], eps)
                        return is_on_point

                    self.dirichlet_bcs.append(fe.DirichletBC(pV, p_val, on_point, method='pointwise'))

            elif bc_type == 'neumann':
                self.add_to_weakform(p_val*fe.dot(w, n)*self.ds(boundary_id))


