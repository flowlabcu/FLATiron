import matplotlib.pyplot as plt
from flatiron_tk.mesh import RectMesh
from flatiron_tk.physics import SteadyScalarTransport
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver

import flatiron_tk
import dolfinx
import ufl 
import numpy as np

from mpi4py import MPI

import sys

mesh = RectMesh(0.0, 0.0, 1.0, 1.0, 1/20)
stp = SteadyScalarTransport(mesh, tag='c')
stp.set_writer('output', 'pvd')
stp.set_element('CG', 1)
stp.build_function_space()

stp.set_advection_velocity([0.0, 0.0])
stp.set_diffusivity(1.0)
stp.set_reaction(0.0)

stp.set_weak_form()
stp.add_stab()  

bottom_flux = dolfinx.fem.Constant(mesh.msh, dolfinx.default_scalar_type([0.0, -0.5]))
top_flux = dolfinx.fem.Constant(mesh.msh, dolfinx.default_scalar_type([0.0, 0.0]))
zero_scalar = flatiron_tk.constant(mesh, 0.0)

bc_dict = {
    1: {'type': 'dirichlet', 'value': zero_scalar},
    2: {'type': 'neumann', 'value': bottom_flux},
    3: {'type': 'dirichlet', 'value': zero_scalar},
    4: {'type': 'neumann', 'value': top_flux},
}

stp.set_bcs(bc_dict)

problem = NonLinearProblem(stp)
solver = NonLinearSolver(mesh.msh.comm, problem)

solver.solve()
stp.write()

