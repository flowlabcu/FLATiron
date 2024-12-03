"""
Solves the benchmark problem from:
https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

from flatiron_tk.physics import SteadyIncompressibleNavierStokes
from flatiron_tk.io import h5_mod
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import PhysicsSolver
import fenics as fe
from flatiron_tk.physics import StokesFlow

# Build mesh
ne = 64
RM = fe.UnitSquareMesh(ne, ne, 'crossed')
mesh = Mesh(mesh=RM)
def left(x, left_bnd):
    return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
def right(x, right_bnd):
    return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
def top(x, top_bnd):
    return abs(top_bnd - x[1]) < fe.DOLFIN_EPS
def bottom(x, bottom_bnd):
    return abs(x[1] - bottom_bnd) < fe.DOLFIN_EPS
mesh.mark_boundary(1, left, (0.))
mesh.mark_boundary(2, bottom, (0.))
mesh.mark_boundary(3, right, (1.))
mesh.mark_boundary(4, top, (1.))

# Define nse equation
Re = 100
mu = 1/Re
rho = 1
nse = SteadyIncompressibleNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()

# Set parameters
nse.set_density(rho)
nse.set_dynamic_viscosity(mu)

# Set weak form
nse.set_weak_form()
nse.add_stab()

# Boundary condition
zero_v = fe.Constant( (0,0) )
zero = fe.Constant(0)
u_bcs = {
        1: {'type': 'dirichlet', 'value': zero_v},
        2: {'type': 'dirichlet', 'value': zero_v},
        3: {'type': 'dirichlet', 'value': zero_v},
        4: {'type': 'dirichlet', 'value': fe.Constant((1, 0))},
        }
p_bcs = {}
bc_dict = {'u': u_bcs,
           'p': p_bcs}
nse.set_bcs(bc_dict)
def bottom_left_corner(x, on_boundary):
    return fe.near(x[0], 0.) and fe.near(x[1], 0.)
nse.dirichlet_bcs.append(fe.DirichletBC(nse.V.sub(1), fe.Constant(0), bottom_left_corner, method='pointwise'))

# Set output writer
nse.set_writer("output", "pvd")

# Solve and write result
la_solver = fe.LUSolver()
solver = PhysicsSolver(nse, la_solver)
solver.solve()
nse.write()

