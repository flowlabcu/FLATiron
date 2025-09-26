import dolfinx 
import numpy as np
import ufl 

from flatiron_tk.fem import boundary_conditions as bcs
from flatiron_tk.physics import MultiphysicsProblem
from flatiron_tk.physics import PhysicsProblem

class Momentum(PhysicsProblem):
    def set_weak_form(self):
        """"""
        
    def flux(self):
        """"""

    def get_residual(self):
        """"""

class Continuity(PhysicsProblem):
    def set_weak_form(self):
        """"""
        
    def flux(self):
        """"""

    def get_residual(self):
        """"""

class OnPoint:
    """
    A class to define a point in space for applying boundary conditions.
    """
    def __init__(self, point_location, eps):
        """
        Parameters:
        -------------
            point_location: The location of the point in space.
            eps: The tolerance for finding the point.
        """
        self.point_location = np.array(point_location)
        self.eps = eps

    def __call__(self, x):
        """
        A method to check if a point is within the tolerance of the defined point location.
        
        Parameters:
        -------------
            x: The coordinates of the point to check.
        
        Returns:
        -------------
            A boolean array indicating if the point is within the tolerance.
        """
        dist = np.linalg.norm(x.T - self.point_location, axis=1)
        return dist < self.eps
    
class SteadyStokes(MultiphysicsProblem):
    """
    A class to represent the steady Stokes equations. Supers MultiphysicsProblem.
        
    Parameters:
    -------------
        mesh: The mesh to use for the problem.
    """
    def __init__(self, mesh):
        self.momentum = Momentum(mesh, tag='u')
        self.continuity = Continuity(mesh, tag='p')
        
        super().__init__(self.momentum, self.continuity)

        self.set_external_function('body_force', dolfinx.fem.Constant(mesh.msh, 
                                                                      dolfinx.default_scalar_type(
                                                                          [0 for i in range(mesh.get_gdim())]
                                                                          )))
        
    def set_element(self, u_family, u_degree, p_family, p_degree):
        """
        Set the element for the Stokes problem.
        
        Parameters:
        -------------
            u_family: The family of the velocity element.
            u_degree: The degree of the velocity element.
            p_family: The family of the pressure element.
            p_degree: The degree of the pressure element.
            mesh: The mesh to use.
        """
        self.momentum.set_element(u_family, u_degree, element_shape='vector')
        
        self.continuity.set_element(p_family, p_degree)
        super().set_element()

    def build_function_space(self):
        """
        Build the function space for the Stokes problem.
        """
        super().build_function_space()

    def set_weak_form(self, stab=False):
        """
        Set the weak form for the Stokes problem.
        This includes the momentum and continuity equations.

        Parameters:
        -----------
            stab: A boolean indicating whether to include stabilization terms in the weak form.
        """

        nu = self.external_function('kinematic_viscosity')
        body_force = self.external_function('body_force')
        dx = self.dx
        u = self.get_solution_function('u')
        p = self.get_solution_function('p')
        w = self.get_test_function('u')
        q = self.get_test_function('p')

        self.weak_form = self._get_stokes_momentum_form(u, p, body_force, nu)
        continuity = q * ufl.div(u) * dx
        self.weak_form += continuity

        if stab:
            self.add_stab()

    def _get_stokes_momentum_form(self, u, p, body_force, nu):
        """
        Define the weak form of the momentum equation for the Stokes problem.
        
        Parameters:
        -------------
            u: The velocity solution function.
            p: The pressure solution function.
            body_force: The body force function.
            nu: The kinematic viscosity function.
        
        Returns:
        -------------
            The weak form of the momentum equation.
        """
        dx = self.dx
        w = self.get_test_function('u')

        a = nu * ufl.inner(ufl.grad(u), ufl.grad(w)) * dx
        b = ufl.div(w) * p * dx
        fb = ufl.inner(body_force, w) * dx

        return a - b - fb
    
    def get_stabilization_constant(self):
        """
        Compute the stabilization constant for the Stokes problem.
        
        Returns:
        -------------
            The stabilization constant.
        """
        h = self.mesh.get_cell_diameter()
        nu = self.external_function('kinematic_viscosity')
        tau = 1./3.*h**2/4/nu
        return tau
    
    def add_stab(self):
        """
        Add stabilization terms to the weak form for the Stokes problem.
        """
        p = self.get_solution_function('p')
        q = self.get_test_function('p')
        P = ufl.grad(q)
        R = ufl.grad(p)
        tau = self.get_stabilization_constant()
        stab_form = tau * ufl.inner(P, R) * self.dx 
        self.add_to_weak_form(stab_form)

    def flux(self, h):
        """
        Computes the flux of the Stokes problem.
        
        Parameters:
        -------------
            h: The flux vector.
            physics_tag: The tag to identify the physics problem.
        
        Returns:
        -------------
            The flux form.
        """
        w = self.get_test_function('w')
        n = self.mesh.get_facet_normal()
        I = ufl.Identity(self.dim)
        flux_form = ufl.dot(w, -ufl.dot(h, n))
        return flux_form

    def set_body_force(self, body_force):
        """
        Set the body force for the Stokes problem.
        
        Parameters:
        -------------
            body_force: The body force value or function.
        """
        self.set_external_function('body_force', body_force)

    def set_kinematic_viscosity(self, nu):
        """
        Set the kinematic viscosity for the Stokes problem.
        
        Parameters:
        -------------
            nu: The kinematic viscosity value or function.
        """
        self.set_external_function('kinematic_viscosity', nu)

    def get_kinematic_viscosity(self):
        """
        Get the kinematic viscosity for the Stokes problem.
        
        Returns:
        -------------
            The kinematic viscosity value or function.
        """
        return self.external_function('kinematic_viscosity')
    
    def set_bcs(self, multiphysics_bc_dict):
        """
        Overload the set_bcs from the multphysics problem to set the boundary conditions for the Stokes problem.
        The boundary conditions on pressure must be handled separately.

        Parameters:
        -----------
            multiphysics_bc_dict: A dictionary containing the boundary conditions for each physics problem.
        """
        u_bcs = multiphysics_bc_dict.get('u', {})
        p_bcs = multiphysics_bc_dict.get('p', {})
        super().set_bcs({'u': u_bcs})

        # Pressure boundary conditions
        n = self.mesh.get_facet_normal()
        w = self.get_test_function('u')
        pV = self.get_function_space('p')

        for boundary_id in p_bcs:
            bc_data = p_bcs[boundary_id]
            bc_type = bc_data['type']
            bc_value = bc_data['value']

            if bc_type == 'dirichlet':

                if isinstance(boundary_id, int):
                    bc = bcs.build_dirichlet_bc(self.mesh, boundary_id, bc_value, self.get_function_space('p'))
                    self.dirichlet_bcs.append(bc)


                elif isinstance(boundary_id, str) and boundary_id.startswith('point'):
                    # Get the function space and dof coordinates
                    pV_base = pV.collapse()[0]
                    p_dof_coords = pV_base.tabulate_dof_coordinates()
                    comm = self.mesh.msh.comm
                    
                    # Ensure 3D coordinate if needed:
                    point_location = bc_data.get('x', [0.0, 0.0])
                    point_location = np.array(point_location)
                    if point_location.size == 2:
                        point_location = np.append(point_location, 0.0)
                    eps = bc_data.get('eps', 1e-6)


                    distances = np.linalg.norm(p_dof_coords - point_location, axis=1)
                    local_min_idx = np.argmin(distances)
                    local_min_dist = distances[local_min_idx]

                    all_min_data = comm.gather((local_min_dist, local_min_idx, comm.rank), root=0)

                    if comm.rank == 0:
                        # Choose the global closest DOF
                        global_min_dist, global_min_idx, owner_rank = min(all_min_data, key=lambda x: x[0])
                    else:
                        global_min_idx = None
                        owner_rank = None

                    if comm.rank == owner_rank:
                        bc = dolfinx.fem.dirichletbc(bc_value, [global_min_idx], pV)
                        self.dirichlet_bcs.append(bc)

            elif bc_type == 'neumann':
                self.add_to_weak_form(bc_value * ufl.dot(w, n)*self.ds(boundary_id))

