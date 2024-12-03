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

def get_flowrate(nse, u, n, id):
	return fe.assemble(fe.dot(u, n) * nse.ds(id))

# Define fluid parameters
mu = 0.04  # g/cm-s
rho = 1.06 # g/cm^3

# Read in mesh files
mesh_file = 'mesh.h5'
mesh = Mesh(mesh_file=mesh_file)
rank = mesh.comm.rank

# Setup Navier-Stokes 
nse = SteadyIncompressibleNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()
nse.set_density(rho)
nse.set_dynamic_viscosity(mu)
nse.set_weak_form(stab=True)
nse.set_writer("output", "pvd")

# Assign boundary conditions
zero_vector = fe.Constant((0, 0))
zero_scalar = fe.Constant(0)

# Boundary conditions
u_bcs = {
		11: {'type':'dirichlet', 'value':fe.Constant((1.0, 0.0))},
		14: {'type':'dirichlet', 'value':zero_vector}
		}

R1 = 0.5
R2 = 0.75








	
