from .non_linear_problem import NonLinearProblem, NonLinearSolver
from ..info.messages import import_fenics
fe = import_fenics()


class PhysicsSolver():

    def __init__(self, physics, la_solver='default'):

        self.physics = physics
        self.set_problem()
        self.set_la_solver(la_solver)
        self.set_problem_solver()

    def set_problem(self):
        self.problem = NonLinearProblem(self.physics)

    def set_la_solver(self, la_solver):
        if la_solver == 'default':
            self.la_solver = fe.LUSolver()
        else:
            self.la_solver = la_solver

    def set_problem_solver(self):
        self.problem_solver = NonLinearSolver(self.physics.mesh.comm, self.problem, self.la_solver)
        # self.problem_solver = fe.PETScSNESSolver("newtonls")
        # snes = self.problem_solver.snes()
        # snes.setKSP(self.la_solver.ksp())

    def set_nonzero_initial_guess(self, is_nonzero):
        self.problem_solver.parameters['krylov_solver']['nonzero_initial_guess'] = is_nonzero

    def solve(self):
        # return self.problem_solver.solve(self.problem, self.physics.solution.vector())
        return self.problem_solver.solve()


