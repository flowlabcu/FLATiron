import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

from flatiron_tk.physics import SteadyFicDomNSE
from flatiron_tk.mesh import RectMesh
from flatiron_tk.solver import *
from petsc4py import PETSc

import fenics as fe

class Disk(fe.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def eval(self, value, x):
        disk_r = 0.2
        disk_xc = 0.5
        disk_yc = 0.5
        xx = x[0] - disk_xc
        yy = x[1] - disk_yc
        r = np.sqrt(xx**2 + yy**2)
        value[0] = 0
        if r <= disk_r:
            value[0] = 1
        return value
    def value_shape(self):
        return ()

dx = 1e-2
mesh = RectMesh(0, 0, 1, 1, dx)

nse = SteadyFicDomNSE(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()
I = Disk()

Re = 100
mu = 1/Re
rho = 1

nse.set_density(rho)
nse.set_dynamic_viscosity(mu)
nse.set_ficdom_domain(I)
nse.set_weak_form()
nse.add_stab()

zero_v = fe.Constant([0, 0])
inlet = fe.Constant([1, 0])
u_bcs = {
        1: {'type': 'dirichlet', 'value': zero_v},
        2: {'type': 'dirichlet', 'value': zero_v},
        3: {'type': 'dirichlet', 'value': zero_v},
        4: {'type': 'dirichlet', 'value': inlet}
        }
p_bcs = {'point_0': {'type': 'dirichlet', 'value':'zero', 'x': (0., 0.)}}
bc_dict = {'u': u_bcs, 'p': p_bcs}
nse.set_bcs(bc_dict)

eps = fe.DOLFIN_EPS
u_ini = fe.interpolate( fe.Constant( (eps, eps) ), nse.V.sub(0).collapse() )
p_ini = fe.interpolate( fe.Constant(eps), nse.V.sub(1).collapse() )
nse.set_initial_guess(u_ini, p_ini)
for bc in nse.dirichlet_bcs:
    bc.apply(nse.solution.vector())

def set_ksp_u(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setMonitor(ConvergenceMonitor("|----KSP0", verbose=False))
    ksp.setTolerances(max_it=20)
    ksp.pc.setType(PETSc.PC.Type.JACOBI)
    ksp.setUp()

def set_ksp_p(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setMonitor(ConvergenceMonitor("|--------KSP1", verbose=False))
    ksp.setTolerances(max_it=5)
    ksp.pc.setType(PETSc.PC.Type.HYPRE)
    ksp.pc.setHYPREType("boomeramg")
    ksp.setUp()

def set_outer_ksp(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setGMRESRestart(30)
    ksp.setTolerances(rtol=1e-8, atol=1e-10)
    ksp.setMonitor(ConvergenceMonitor("Outer ksp"))

split = {'fields': ('u', 'p'),
         'composite_type': 'schur',
         'schur_fact_type': 'full',
         'schur_pre_type': 'a11',
         'ksp0_set_function': set_ksp_u,
         'ksp1_set_function': set_ksp_p}

nse.set_writer("output", "pvd")
problem = NonLinearProblem(nse)
tree = FieldSplitTree(nse, split)
solver = BlockNonLinearSolver(tree, mesh.comm, problem, 
                              fe.PETScKrylovSolver(), 
                              outer_ksp_set_function=set_outer_ksp)

# solver = BlockNonLinearSolver(tree, mesh.comm, problem, fe.LUSolver())
solver.solve()
nse.write()
