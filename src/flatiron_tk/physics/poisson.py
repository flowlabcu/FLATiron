import ufl

from flatiron_tk.physics import PhysicsProblem

class Poisson(PhysicsProblem):
    """
    A minimal Poisson problem, -div(grad(u)) = f.
    """

    def build_function_space(self, *args, **kwargs):
        """
        Build the function space for the Poisson problem.

        Parameters
        ----------
            *args: Arguments to be passed to the parent class method.
            **kwargs: Keyword arguments to be passed to the parent class method.
        """
        super().build_function_space(*args, **kwargs)
        self.solution.name = self.tag

    def set_source(self, f):
        """
        Set the source term for the Poisson problem.

        Parameters
        -------------
            f: The source term.
        """
        self.set_external_function('source', f)

    def get_source(self):
        """
        Get the source term for the Poisson problem.

        Returns:
        -------------
            The source term.
        """
        return self.external_function('source')

    def set_weak_form(self):
        """
        Set the weak form for the Poisson problem.
        """
        w = self.get_test_function()
        u = self.get_solution_function()
        f = self.get_source()

        self.weak_form = ufl.inner(ufl.grad(w), ufl.grad(u)) * self.dx - w * f * self.dx

    def flux(self, h):
        """
        Define the flux term for Neumann boundary conditions.

        Parameters
        -------------
            h: The flux value.
        """
        w = self.get_test_function()
        return -w * h
