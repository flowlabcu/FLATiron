import numpy as np
import sys
import time
import copy
import os

# ------------------------------------------------------- #
from ..info.messages import import_fenics
from .convergence_monitor import ConvergenceMonitor

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
            self._mpi_comm = comm
            self.problem = problem
            self.ksp_is_initialized = False
            self._outer_ksp_set_func = kwargs.pop("outer_ksp_set_function", self.default_set_ksp)
            fe.NewtonSolver.__init__(self, comm, la_solver, fe.PETScFactory.instance())

        def init_ksp(self):

            """
            This step fine tunes the underlying KSP object of the linear solver.
            Note: You don't need to call this method yourself. This method will
            be called in the solver_setup step, where we already handle the fact
            that it can only be called once through the `ksp_is_initialized` variable
            """
            ksp = self.linear_solver().ksp()
            self._outer_ksp_set_func(ksp)

        def default_set_ksp(self, ksp):
            """
            Set default values to be a direct solve.
            Here I am adding the settings explicitly here
            as a hint for future users if they want to edit this bit
            """
            ksp.setType(ksp.Type.FGMRES)
            ksp.pc.setType(ksp.pc.Type.JACOBI)
            ksp.setGMRESRestart(30)
            ksp.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
            ksp.setMonitor(ConvergenceMonitor('ksp'))

        def solver_setup(self, A, P, problem, iteration):
            """
            This method is called by FEniCS everytime we solve an iteration within
            the Newton iteration. We will set the operator to A every time, but
            the KSP initialization will only be done once
            """
            self.linear_solver().set_operator(A)

            # Make sure init_ksp is only called once
            if (not self.ksp_is_initialized
                and not isinstance(self.linear_solver(), fe.cpp.la.LUSolver)):
                self.init_ksp()
                self.ksp_is_initialized = True

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

