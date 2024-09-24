import numpy as np
import sys
import time
import copy
import os

# ------------------------------------------------------- #
from ..info.messages import import_fenics
fe = import_fenics()

class NonLinearProblem():
    pass

class NonLinearSolver():
    pass

if fe:

    class NonLinearProblem(fe.NonlinearProblem):

        def __init__(self, physics):

            # Grab weakform and its derivative from the physics class
            self.physics = physics
            F = physics.get_weak_form()
            sol = physics.solution
            J = fe.derivative(F, sol)

            # Define it back into the nonlinear problem class
            self.bilinear_form = J
            self.linear_form = F
            self.bcs = physics.dirichlet_bcs
            fe.NonlinearProblem.__init__(self)

        def F(self, b, x):
            fe.assemble(self.linear_form, tensor=b)
            for bc in self.bcs:
                bc.apply(b, x)

        def J(self, A, x):
            fe.assemble(self.bilinear_form, tensor=A)
            for bc in self.bcs:
                bc.apply(A)

    class NonLinearSolver(fe.NewtonSolver):

        def __init__(self, comm, problem, la_solver, **kwargs):
            self.problem = problem
            fe.NewtonSolver.__init__(self, comm, la_solver, fe.PETScFactory.instance())

        def solve(self):
            sol_vector = self.problem.physics.solution
            super().solve(self.problem, sol_vector.vector())


# class NonLinearSolver(fe.NewtonSolver):

#     def __init__(self, comm, problem, la_solver, **kwargs):
#         self.problem = problem
#         self.solver_type = kwargs.pop('solver_type', 'gmres')
#         self.pc_type     = kwargs.pop('pc_type'    , 'hypre')
#         self.rel_tol     = kwargs.pop('relative_tolerance', 1e-2)
#         self.abs_tol     = kwargs.pop('absolute_tolerance', 1e-3)
#         self.max_iter    = kwargs.pop('maximum_iterations', 1000)
#         fe.NewtonSolver.__init__(self, comm,
#                               fe.PETScKrylovSolver(), fe.PETScFactory.instance())

#     def solver_setup(self, A, P, problem, iteration):
#         self.linear_solver().set_operator(A)
#         fe.PETScOptions.set("ksp_type", self.solver_type)
#         fe.PETScOptions.set("ksp_monitor")
#         fe.PETScOptions.set("pc_type", self.pc_type)
#         self.linear_solver().parameters["relative_tolerance"] = self.rel_tol
#         self.linear_solver().parameters["absolute_tolerance"] = self.abs_tol
#         self.linear_solver().parameters["maximum_iterations"] = self.max_iter
#         self.linear_solver().set_from_options()

#     def solve(self):
#         sol_vector = self.problem.physics.solution
#         super().solve(self.problem, sol_vector.vector())

