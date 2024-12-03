import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

from flatiron_tk.physics import FicDomNSE
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
mesh = RectMesh(0, 0, 6, 1, dx)
nse = FicDomNSE(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()
I = Disk()
QE = fe.FiniteElement("Quadrature", mesh.fenics_mesh().ufl_cell(), 4, quad_scheme="default")
VE = fe.FunctionSpace(mesh.fenics_mesh(), QE)
I = fe.interpolate(I, VE)

Re = 100
U = 1
rho = 1
mu = 1/Re
CFL = 1
dt = CFL*dx/U

nse.set_density(rho)
nse.set_dynamic_viscosity(mu)
nse.set_ficdom_domain(I)
nse.set_time_step_size(dt)
nse.set_mid_point_theta(0.5)
nse.set_weak_form()
nse.add_stab()

zero_v = fe.Constant([0, 0])
zero = fe.Constant(0)
inlet = fe.Expression( ("4*U*x[1]*(1-x[1])", "0"), U=U, degree=2 ) # times 4 to make the midpoint velocity = U
u_bcs = {
        1: {'type': 'dirichlet', 'value': inlet},
        2: {'type': 'dirichlet', 'value': zero_v},
        4: {'type': 'dirichlet', 'value': zero_v}
        }
p_bcs = {3: {'type': 'dirichlet', 'value': zero}}
bc_dict = {'u': u_bcs, 'p': p_bcs}
nse.set_bcs(bc_dict)

def set_ksp_u(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setMonitor(ConvergenceMonitor("|----KSP0", verbose=False))
    ksp.setTolerances(max_it=5)
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
    ksp.setMonitor(ConvergenceMonitor("Outer ksp", verbose=False))

split = {'fields': ('u', 'p'),
         'composite_type': 'schur',
         'schur_fact_type': 'full',
         'schur_pre_type': 'a11',
         'ksp0_set_function': set_ksp_u,
         'ksp1_set_function': set_ksp_p}

nse.set_writer("output", "pvd")
problem = NonLinearProblem(nse)
tree = FieldSplitTree(nse, split)
la_solver = fe.PETScKrylovSolver()
# la_solver = fe.LUSolver()
solver = BlockNonLinearSolver(tree, mesh.comm, problem,
                              la_solver,
                              outer_ksp_set_function=set_outer_ksp)


T = 8
nt = int(T/dt)+1
print(nt)

for time_step in range(nt):

    solver.solve()
    nse.update_previous_solution()

    if time_step%1==0:
        nse.write()
