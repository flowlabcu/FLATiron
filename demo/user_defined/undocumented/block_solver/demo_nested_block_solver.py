"""
Solve a simple galerkin projection problem of 3 variables
preconditioned using a nested fieldsplit preconditioner
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os
from flatiron_tk.physics import PhysicsProblem, MultiPhysicsProblem
from flatiron_tk.mesh import LineMesh
from flatiron_tk.solver import NonLinearProblem, NonLinearSolver
from flatiron_tk.solver import BlockNonLinearSolver, FieldSplitTree
import fenics as fe

class GalerkinProjection(PhysicsProblem):
    '''
    GalerkinProjection field_value = b
    '''
    def set_element(self, element_family, element_degree, dim):
        self.element = fe.VectorElement(element_family, self.mesh.mesh.ufl_cell(), element_degree, dim=dim)
        self.element_family = element_family
        self.element_degree = element_degree

    def set_projection_value(self, projection_value):
        self.set_external_function('b', projection_value)

    def set_weak_form(self):
        b = self.external_function('b')
        u = self.solution_function()
        w = self.test_function()
        self.weak_form = fe.dot(u-b, w)*self.dx

def build_GP(tag, mesh, dim, val):
    GP = GalerkinProjection(mesh, tag)
    GP.set_element('CG', 1, dim)
    GP.set_projection_value(fe.Constant(val))
    return GP

mesh = LineMesh(0, 1, 1/10)
GP1 = build_GP('A', mesh, dim=2, val=[1,2])
GP2 = build_GP('B', mesh, dim=3, val=[3,4,5])
GP3 = build_GP('C', mesh, dim=4, val=[6,7,8,9])
GPs = [GP1, GP2, GP3]

physics = MultiPhysicsProblem(*GPs)
physics.set_element()
physics.build_function_space()
physics.set_weak_form()

split0 = {'fields': (('A','C'),'B'),
          'composite_type': 'schur',
          'schur_fact_type': 'full',
          'schur_pre_type': 'a11'}

split1 = {'fields': ('A','C'),
          'composite_type': 'schur',
          'schur_fact_type': 'full',
          'schur_pre_type': 'a11'}

splits = [split0, split1]

tree = FieldSplitTree(physics, splits)

problem = NonLinearProblem(physics)
solver = BlockNonLinearSolver(tree, fe.MPI.comm_world, problem, fe.PETScKrylovSolver())
solver.solve()
(A, B, C) = physics.solution.split(True)
print('A', A.vector()[:])
print('B', B.vector()[:])
print('C', C.vector()[:])




