from flatiron_tk.physics import PhysicsProblem
from flatiron_tk.physics import MultiphysicsProblem
from flatiron_tk.physics import SteadyStokes
import numpy as np

import dolfinx 
import ufl 

class SteadyNavierStokes(SteadyStokes):
    """
    A class to represent the steady Navier-Stokes equations.
    This class extends the SteadyStokes class to include the convective term
    in the momentum equation.
    """

    def set_initial_guess(self, initial_guess_u=None, initial_guess_p=None):
        """
        Set the initial guess for the steady Navier-Stokes problem.
        
        Parameters:
        ------------
            initial_guess_u: A user-defined initial guess for the velocity field. If None, a zero initial guess is used.
            initial_guess_p: A user-defined initial guess for the pressure field. If None, a zero initial guess is used.
        
        Returns:
        -----------
            None
        """
        u_sol = self.get_solution_function('u')
        p_sol = self.get_solution_function('p')

        if initial_guess_u is not None:
            # User provided velocity initial guess
            if isinstance(initial_guess_u, dolfinx.fem.Function):
                u_sol.x.array[:] = initial_guess_u.x.array
            else:
                u_sol.interpolate(initial_guess_u)
        else:
            # Zero initial guess
            u_sol.interpolate(lambda x: np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=dolfinx.default_scalar_type))

        if initial_guess_p is not None:
            # User provided pressure initial guess
            if isinstance(initial_guess_p, dolfinx.fem.Function):
                p_sol.x.array[:] = initial_guess_p.x.array
            else:
                p_sol.interpolate(initial_guess_p)
        else:
            # Zero initial guess
            p_sol.interpolate(lambda x: np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type))

    def set_weak_form(self, stab=False):
        """
        Set the weak form for the steady Navier-Stokes problem.
        
        Parameters:
        -----------
            stab: A boolean indicating whether to include stabilization terms in the weak form.
        """
        V = self.V

        mu = self.external_function('dynamic_viscosity')
        rho = self.external_function('density')
        body_force = self.external_function('body_force')
        dx = self.dx
        u = self.get_solution_function('u')
        p = self.get_solution_function('p')
        w = self.get_test_function('u')
        q = self.get_test_function('p')

        T0 = rho * ufl.inner(w, ufl.grad(u)*u) * dx 
        T1 = mu * ufl.inner(ufl.grad(w), ufl.grad(u)) * dx 
        T2 = p * ufl.div(w) * dx
        T3 = q * ufl.div(u) * dx
        T4 = rho * ufl.dot(w, body_force) * dx

        self.weak_form = T0 + T1 - T2 + T3 - T4 

        if stab:
            self.add_stab()

    def add_stab(self):
        """
        Add stabilization terms to the weak form for the steady Navier-Stokes problem.
        """
        u = self.get_solution_function('u')
        p = self.get_solution_function('p')
        w = self.get_test_function('u')
        q = self.get_test_function('p')
        rho = self.external_function('density')
        h = self.mesh.get_cell_diameter()

        r = self.get_residual(u, p)
        (tau_SUPG, tau_PSPG) = self.get_stab_constant(u, h, 1.0, 1.0)
        stab_SUPG = tau_SUPG*ufl.inner(ufl.grad(w)*u, r)
        stab_PSPG = 1/rho * tau_PSPG * ufl.inner(ufl.grad(q), r)

        self.add_to_weak_form(stab_SUPG, self.dx)
        self.add_to_weak_form(stab_PSPG, self.dx)

    def get_residual(self, u, p):
        """
        Compute the residual for the steady Navier-Stokes problem.
        
        Parameters:
        -----------
            u: The velocity field.
            p: The pressure field.
        
        Returns:
        -----------
            The residual of the steady Navier-Stokes equations.
        """
        mu = self.external_function('dynamic_viscosity')
        rho = self.external_function('density')
        body_force = self.external_function('body_force')

        return rho*ufl.grad(u)*u - mu*ufl.div(ufl.grad(u)) + ufl.grad(p) - rho*body_force
        
    
    def get_stab_constant(self, u, h, alpha, beta):
        """
        Compute the stabilization constants for the steady Navier-Stokes problem.
        
        Parameters:
        -------------
            u: The velocity field.
            h: The cell diameter.
            alpha: The SUPG stabilization parameter.
            beta: The PSPG stabilization parameter.
        
        Returns:
        -------------
            A tuple containing the SUPG and PSPG stabilization constants.
        """
        return self._T_SUPG(u, h, alpha), self._T_PSPG(u, h, beta)

    def _T_SUPG(self, u, h, alpha):
        """
        Compute the SUPG stabilization term for the steady Navier-Stokes problem.
        
        Parameters:
        -------------
            u: The velocity field.
            h: The cell diameter.
            alpha: The SUPG stabilization parameter.
        
        Returns:
        -------------
            The SUPG stabilization term.
        """
        mu = self.external_function('dynamic_viscosity')
        rho = self.external_function('density')
        u2 = ufl.dot(u,u)
        nu = mu/rho
        return alpha * ( (4*u2)/h**2 + 9*(4*nu/h**2)**2 ) ** (-0.5)

    def _T_PSPG(self, u, h, beta):
        """
        Compute the PSPG stabilization term for the steady Navier-Stokes problem.
        
        Parameters:
        -------------
            u: The velocity field.
            h: The cell diameter.
            beta: The PSPG stabilization parameter.
        
        Returns:
        -------------
            The PSPG stabilization term.
        """
        mu = self.external_function('dynamic_viscosity')
        rho = self.external_function('density')
        u2 = ufl.dot(u,u)
        nu = mu/rho
        return beta * ( (4*u2)/h**2 + 9*(4*nu/h**2)**2 ) ** (-0.5)

    def set_dynamic_viscosity(self, mu):
        """
        Set the dynamic viscosity for the Navier-Stokes problem.
        
        Parameters:
        -------------
            mu: The dynamic viscosity value.
        """
        self.set_external_function('dynamic_viscosity', mu)

    def set_density(self, rho):
        """
        Set the density for the Navier-Stokes problem.
        
        Parameters:
        -------------
            rho: The density value.
        """
        self.set_external_function('density', rho)