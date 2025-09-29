from flatiron_tk.info import *
petsc = import_dolfinx_fem_petsc()

class NonLinearProblem(petsc.NonlinearProblem):
    """
    Non linear problem class for solving nonlinear PDEs using the
    dolfinx nonlinear solver. Supers the petsc.NonlinearProblem class.
    
    Parameters
    -----------
    physics: 
    The physics object problem to be solved.
    """
    
    def __init__(self, physics):
        self.physics = physics
        
        self.weak_form = physics.get_weak_form()
        self.jacobian = physics.jacobian()
        self.solution_function = physics.get_solution_function()

        super().__init__(
                        F=self.weak_form,
                        u=self.solution_function,
                        bcs=physics.dirichlet_bcs,
                        J=self.jacobian
                        )