from ..info.messages import import_fenics
fe = import_fenics()
from .physics_problem import PhysicsProblem
from ..io import h5_mod
import os
import numpy as np

class ElastoDynamics(PhysicsProblem):
    """
    Solve linear isotropic elastodynamic problem
    """

    def __init__(self, mesh, tag='u', *args, **kwargs):
        r"""
        :param mesh: Mesh domain
        :type mesh: flatiron_tk.Mesh

        :param tag: Tag for this physics
        :type tag: str
        :default tag: 'u'


        """

        # Initialize underlying physics
        super().__init__(mesh, tag=tag, *args, **kwargs)

        # Gen default gen alpha params
        self.set_gen_alpha(alpha_m=0.2, alpha_f=0.4)

        # Default to no damping
        self.set_damping(eta_m=0, eta_k=0)

        # Default external acceleration is 0
        zero_v = fe.Constant([0. for _ in range(self.mesh.dim)])
        self.set_external_acceleration(fe.Constant(zero_v))

    def set_element(self, element_family, element_degree):
        self.element = fe.VectorElement(element_family, self.mesh.fenics_mesh().ufl_cell(), element_degree)

    def build_function_space(self):
        super().build_function_space()
        V = self.function_space()
        u_old = fe.Function(V)
        v_old = fe.Function(V)
        a_old = fe.Function(V)
        self._previous_solutions = {
                'u': u_old,
                'v': v_old,
                'a': a_old
                }

        # Build stress function space
        # The function space degree of stress deg_sigma := deg_u-1
        # where deg_u is the degree of the displacement function space
        # If deg_u = 1, i.e., CG1, then deg_sigma will be DG0
        # This will be the degree of the displacement
        deg_u = V.ufl_element().degree()
        fam_u = V.ufl_element().family()
        self.Vsig_deg = deg_u - 1
        if self.Vsig_deg == 0:
            self.Vsig_family = 'DG'
        else:
            self.Vsig_family = fam_u
        self.Vsig = fe.TensorFunctionSpace(self.mesh.fenics_mesh(), self.Vsig_family, self.Vsig_deg)

    def previous_solution(self, name):
        return self._previous_solutions[name]

    def set_external_acceleration(self, aext):
        self.set_external_function('external acceleration', aext)

    def set_density(self, rho):
        '''
        Structure's mass density
        '''
        self.set_external_function('density', rho)

    def set_dt(self, dt):
        self.set_external_function('dt', dt)

    def set_gen_alpha(self, alpha_m, alpha_f):
        r'''
        Set the generalized alpha parameters alpha_m and alpha_f

        :param alpha_m: constant describing the mid-point on the acceleration term :math:`a_{mid}=(\alpha_m*a_{old} + (1-\alpha_m)*a_{new})`
        :type lmbda: float

        :param alpha_f: constant describing the mid-point on the displacement and velocity term :math:`u_{mid}=(\alpha_f*u_{old} + (1-\alpha_f)*u_{new})`
        :type alpha_f: float

        '''
        gamma = 0.5 + alpha_f - alpha_m
        beta = (gamma+0.5)**2/4.
        self.set_external_function('gen alpha alpha_m', alpha_m)
        self.set_external_function('gen alpha alpha_f', alpha_f)
        self.set_external_function('gen alpha gamma', gamma)
        self.set_external_function('gen alpha beta', beta)

    def set_lames_const(self, lmbda, mu):
        r'''
        Lame's constant for linear elasticity such that :math:`\sigma = \lambda tr(\epsilon(\textbf{u}))I + 2\mu\epsilon(\textbf{u})`

        :param lmbda: First Lame's constant (:math:`\lambda`)
        :type lmbda: float

        :param mu: Second Lame's constant (:math:`\mu`)
        :type mu: float
        '''
        self.set_external_function('lmbda', lmbda)
        self.set_external_function('mu', mu)

    def set_damping(self, eta_m, eta_k):
        r'''
        Sets raleigh damping coefficients defined as
        C(\textbf{v}, \textbf{w}) = \eta_m * M(\textbf{v}, \textbf{w}) + \eta_k * K(\textbf{v}, \textbf{w})
        where M(\textbf{v}, \textbf{w}) and K(\textbf{v}, \textbf{w}) are the mass and stiffness
        matrix respectively
        '''
        self.set_external_function('eta_m', eta_m)
        self.set_external_function('eta_k', eta_k)

    def stress(self, u):
        r'''
        Returns the linear elastic stress tensor :math:`\sigma(\textbf{u})`

            .. math::

                \sigma(\textbf{u}) = \lambda tr\left(\epsilon(\textbf{u})\right) + 2\mu\epsilon(\textbf{u})

        where :math:`\epsilon(\textbf{u})` is the symmetric strain tensor

            .. math::

                \epsilon(\textbf{u}) = \frac{1}{2}\left(\nabla{\textbf{u}} + \nabla{\textbf{u}}^T\right)
        '''
        mu = self.external_function('mu')
        lmbda = self.external_function('lmbda')
        eps = fe.sym(fe.grad(u))
        I = fe.Identity(len(u))
        sigma = lmbda*fe.tr(eps)*I + 2.0*mu*eps
        return sigma

    def M(self, u, w):
        r'''
        Returns the mass term :math:`M(\textbf{u}, \textbf{w})`

            .. math::

                M(\textbf{u}, \textbf{w}) = \rho \left(\textbf{u}, \textbf{w} \right)_{\Omega}


        '''
        rho = self.external_function('density')
        return rho*fe.inner(u, w)

    def K(self, u, w):
        r'''
        Returns the Elastic term :math:`K(\textbf{u}, \textbf{w})`

            .. math::

                K(\textbf{u}, \textbf{w}) = \left( \sigma(\textbf{u}), \epsilon(\textbf{w}) \right)_{\Omega}

        '''
        sigma = self.stress(u)
        return fe.inner(sigma, fe.sym(fe.grad(w)))

    def C(self, v, w):
        r'''
        Returns the Raleigh damping term :math:`C(\textbf{v}, \textbf{w})`

            .. math::

                C(\textbf{v}, \textbf{w}) = \eta_m M(\textbf{v}, \textbf{w}) + \eta_kK(\textbf{v}, \textbf{w})

        '''
        eta_m = self.external_function('eta_m')
        eta_k = self.external_function('eta_k')
        return eta_m*self.M(v, w) + eta_k*self.K(v, w)

    def L(self, w):
        r'''
        eturns the external force term :math:`L(\textbf{w})`

            .. math::

                L(\textbf{w}) = \left(\rho\textbf{a}_{ext}, \textbf{w}\right)_\Omega

        '''
        a_ext = self.external_function('external acceleration')
        rho = self.external_function('density')
        return fe.dot(rho*a_ext, w)


    def _mid(self, x0, x1, alpha):
        r'''
        This method computes :math:`x_{mid} = \alpha x_{0} + (1-\alpha) x_{1}

        '''
        return alpha*x0 + (1-alpha)*x1

    def _get_float_value(self, fe_const):
        r'''
        Get value from a fe.Constant object
        The input fe.Constant object Must be a scalar value

        '''
        assert( isinstance(fe_const, fe.Constant) )
        return fe_const.values()[0]

    def _get_a(self, u, u_old, v_old, a_old, as_float=False):
        dt = self.external_function('dt')
        beta = self.external_function('gen alpha beta')
        if as_float:
            dt_ = self._get_float_value(dt)
            beta_ = self._get_float_value(beta)
        else:
            dt_ = dt
            beta_ = beta
        a = (u - u_old - dt_*v_old)/beta_/dt_**2 - (1 - 2*beta_)/2/beta_*a_old
        return a

    def _get_v(self, a, u_old, v_old, a_old, as_float=False):
        dt = self.external_function('dt')
        gamma = self.external_function('gen alpha gamma')
        if as_float:
            dt_ = self._get_float_value(dt)
            gamma_ = self._get_float_value(gamma)
        else:
            dt_ = dt
            gamma_ = gamma
        v = v_old + dt_*((1-gamma_)*a_old + gamma_*a)
        return v

    def _gen_alpha_update(self, u, u_old, v_old, a_old):

        # Get vectors (references)
        u_vec, u0_vec  = u.vector(), u_old.vector()
        v0_vec, a0_vec = v_old.vector(), a_old.vector()

        # use update functions using vector arguments
        a_vec = self._get_a(u_vec, u0_vec, v0_vec, a0_vec, as_float=True)
        v_vec = self._get_v(a_vec, u0_vec, v0_vec, a0_vec, as_float=True)

        # Update (u_old <- u)
        v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
        u_old.vector()[:] = u.vector()

    def set_weak_form(self):

        r"""
        self.weak_form = :math:`M(\textbf{a}_{mid}, \textbf{w}) + C(\textbf{v}_{mid}, \textbf{w}) + K(\textbf{u}_{mid}, \textbf{w}) - L(\textbf{w})`
        """

        alpha_m = self.external_function('gen alpha alpha_m')
        alpha_f = self.external_function('gen alpha alpha_f')
        u_old = self.previous_solution('u')
        v_old = self.previous_solution('v')
        a_old = self.previous_solution('a')

        u = self.solution_function()
        w = self.test_function()

        a_new = self._get_a(u, u_old, v_old, a_old)
        v_new = self._get_v(a_new, u_old, v_old, a_old)

        a_mid = self._mid(a_old, a_new, alpha_m)
        v_mid = self._mid(v_old, v_new, alpha_f)
        u_mid = self._mid(u_old, u, alpha_f)

        self.weak_form = self.M(a_mid, w) + self.C(v_mid, w) + self.K(u_mid, w) - self.L(w)
        self.weak_form *= self.dx

    def add_traction(self, p, face_id):
        r'''
        Apply work due to external force on a face face_id
        '''
        w = self.test_function()
        self.weak_form -= fe.dot(w, p)*self.ds(face_id)

    def update_previous_solution(self):
        u_old = self.previous_solution('u')
        v_old = self.previous_solution('v')
        a_old = self.previous_solution('a')
        u = self.solution_function()
        self._gen_alpha_update(u, u_old, v_old, a_old)

    def flux(self, t=None, sigma=None):
        r'''
        The "flux" here represents the traction being applied on a face of the structure
        we have two type of flux that the user can supply.
        Let :math:`\textbf{w}` be the test function
        If traction force is supplied:
        return :math:`-\textbf{w} \cdot \textbf{t}`
        If surface stress is supplied:
        return :math:`-\textbf{w} \cdot sigma \cdot \textbf{n}`
        where n is the face normal vector


        '''

        traction_is_supplied = t is not None
        stress_is_supplied = sigma is not None

        if traction_is_supplied and stress_is_supplied:
            raise ValueError("Both traction and stress is supplied. Please only supply one or the other")

        if not traction_is_supplied and not stress_is_supplied:
            raise ValueError("Please supply either a traction or a stress")

        w = self.test_function()
        if stress_is_supplied:
            n = self.mesh.facet_normal()
            t = fe.dot(sigma, n)
        return -fe.dot(w, t)

    def set_bcs(self, bcs_dict):
        super().set_bcs(bcs_dict)
        for boundary_id in bcs_dict:
            bc_data = bcs_dict[boundary_id]
            bc_value = bc_data['value']
            if bc_data['type'] == 'neumann traction':
                flux_term = self.flux(traction=bc_value)
            elif bc_data['type'] == 'neumann stress':
                flux_term = self.flux(stress=bc_value)
            else:
                continue
            self.add_to_weakform(flux_term, self.ds(boundary_id))

    def get_residue(self):
        pass

    def set_writer(self, directory, file_format, write_stress=False):

        r"""
        :param directory: Directory to write the data
        :type directory: str
        :param file_format: Either 'h5' or 'pvd'
        :type file_format: str
        :param write_stress: Whether to project and write the stress field
        :type write_stress: bool
        :default write_stress: False
        """
        self._write_stress = write_stress

        # Set directory
        os.system("mkdir -p %s" % directory)

        # Set outputfile name and field name
        # self.output_file = os.path.join(directory, self.tag+'.'+file_format)
        output_quantities = ['displacement', 'velocity', 'acceleration', 'stress']
        output_file_name = [ os.path.join(directory, self.tag+'_'+field+'.'+file_format)
                for field in output_quantities ]
        self.output_file = dict(zip(output_quantities, output_file_name))

        # Set output fid
        if file_format == 'h5':
            output_fid = [self._get_h5_file(f) for f in output_file_name]
        elif file_format == 'pvd':
            output_fid = [self._get_pvd_file(f) for f in output_file_name]
        else:
            raise ValueError("Incorrect output file extension")
        self.output_fid = dict(zip(output_quantities, output_fid))
        self.file_format = file_format

        # Print out the path
        if self.mesh.comm.rank == 0:
            print("Output file set to %s"%self.output_file)

    def project_stress(self):
        '''
        Project the UFL form of the stress term to the mesh

        '''
        u = self.previous_solution('u')
        sigma = self.stress(u)
        sigma = fe.project(sigma, self.Vsig)
        return sigma

    def write(self, **kwargs):

        # Get time stamp if applicable
        time_stamp = kwargs.pop("time_stamp", self.number_of_steps_written)
        self.number_of_steps_written += 1

        u = self.previous_solution('u')
        v = self.previous_solution('v')
        a = self.previous_solution('a')

        file_format = self.file_format
        self._write_function(file_format, 'displacement', u,     time_stamp)
        self._write_function(file_format, 'velocity',     v,     time_stamp)
        self._write_function(file_format, 'acceleration', a,     time_stamp)


        if self._write_stress:
            sigma = self.project_stress()
            self._write_function(file_format, 'stress', sigma, time_stamp)

    def _write_function(self, fformat, fname, function, time_stamp):
        assert(fformat=='h5' or fformat=='pvd')
        fid = self.output_file[fname]
        fid = self.output_fid[fname]
        if fformat == 'h5':
            h5_mod.h5_write(function, fname, h5_object=fid, timestamp=time_stamp)
        else:
            function.rename(fname, fname)
            fid.write(function, time_stamp)

