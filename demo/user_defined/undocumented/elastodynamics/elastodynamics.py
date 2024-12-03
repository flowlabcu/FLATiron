import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

import fenics as fe
from flatiron_tk.mesh import BoxMesh
# from flatiron_tk.physics import PhysicsProblem
from flatiron_tk.solver import PhysicsSolver
from flatiron_tk.io import h5_mod
from flatiron_tk.physics import ElastoDynamics


"""
This is the problem setup found in legacy FEniCS demos
Here, we simply re-cast it to FLATiron's code
https://olddocs.fenicsproject.org/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html
"""

# Define mesh
lx = 1
ly = 0.1
lz = 0.04
h = lz/4
mesh = BoxMesh(0, 0, 0, lx, ly, lz, h)

# Time stepping
T = 4
Nsteps = 50
time = np.linspace(0, T, Nsteps+1)
dt = time[1] - time[0]

# Structure's param
rho = 1
E = 1e3
nu = 0.3
mu = E/(2*(1+nu))
lmbda = E*nu/ ( (1+nu)*(1-2*nu) )

# Build physics
eldy = ElastoDynamics(mesh)
eldy.set_density(rho)
eldy.set_dt(dt)
eldy.set_lames_const(lmbda, mu)
eldy.set_element('CG', 1)
eldy.build_function_space()
eldy.set_weak_form()

# Loading vector
p0 = 1.0
cutoff_Tc = T/5
p = fe.Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)

# Boundary condition
zero_v = fe.Constant((0, 0, 0))
bc_dict = {1: {'type': 'dirichlet', 'value': zero_v},
           4: {'type': 'neumann traction', 'value': p}}
eldy.set_bcs(bc_dict)

# Set io
eldy.set_writer("output", "pvd")

# Set linear solver
la_solver = fe.LUSolver()
solver = PhysicsSolver(eldy)

# Solve
alpha_f = eldy.external_function('gen alpha alpha_f').values()[0]
E_damp = 0
energies = np.zeros( (Nsteps+1, 4) )
for i, t in enumerate(time):

    # Update loading vector
    p.t = t - (alpha_f*dt)

    # Solve
    solver.solve()
    eldy.write(time_stamp=t)
    eldy.update_previous_solution()

    # Compute internal energies
    u_old = eldy.previous_solution('u')
    v_old = eldy.previous_solution('v')
    dx = eldy.dx
    E_elas = fe.assemble(0.5*eldy.K(u_old, u_old)*dx)
    E_kin = fe.assemble(0.5*eldy.M(v_old, v_old)*dx)
    E_damp += dt*fe.assemble(eldy.C(v_old, v_old)*dx)
    E_tot = E_elas + E_kin + E_damp
    energies[i, :] = np.array([E_elas, E_kin, E_damp, E_tot])


# Plot diagnostics
if fe.MPI.comm_world.rank == 0:
    plt.plot(time, energies)
    plt.legend(("elastic", "kinetic", "damping", "total"))
    plt.xlabel("Time")
    plt.ylabel("Energies")
    plt.ylim(0, 0.0011)
    plt.show()


