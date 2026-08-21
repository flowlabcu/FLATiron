class LinearProblem:
    """
    Linear problem class for physics whose weak form is exactly linear in
    the solution function (e.g. Poisson, steady diffusion). Unlike
    NonLinearProblem, this does not subclass dolfinx's SNES-based
    NonlinearProblem -- it just holds the pieces LinearSolver needs to do a
    single assemble + single KSP solve, with no re-linearization or
    convergence-check loop.

    Parameters
    -----------
    physics:
        The physics object problem to be solved. Its weak form must be
        linear in the solution function.
    """

    def __init__(self, physics):
        self.physics = physics
        self.weak_form = physics.get_weak_form()
        self.jacobian = physics.jacobian()
        self.solution_function = physics.get_solution_function()
