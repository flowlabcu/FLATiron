import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #
#https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html

import fenics as fe
from flatiron_tk.physics import TransientMultiPhysicsProblem, TransientScalarTransport, IncompressibleNavierStokes
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import PhysicsSolver

# Constants
dt = 0.0025
mu = 0.001
rho = 1

mesh_file = '../../mesh/h5/foc.h5'
mesh = Mesh(mesh_file = mesh_file)

# Set nse
nse = IncompressibleNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.set_time_step_size(dt)
nse.set_mid_point_theta(1)
nse.set_density(rho)
nse.set_dynamic_viscosity(mu)

# Set adr
adr = TransientScalarTransport(mesh, dt, theta=0.5)
adr.tag = 'c'
adr.set_element('CG', 1)
adr.set_diffusivity(mu, mu)
adr.set_reaction(0.0, 0.0)

coupled_physics = TransientMultiPhysicsProblem(nse, adr)
coupled_physics.set_element()
coupled_physics.build_function_space()
u = coupled_physics.solution_function('u')
u0 = fe.split(coupled_physics.sub_physics[0].previous_solution)[0]
# NOTE: Implicit coupling doesn't work... saddle point probably
adr.set_advection_velocity(u0, u0)

nse_options = {'stab': True}
adr_options = {'stab': True}
coupled_physics.set_weak_form(nse_options, adr_options)

# BCs
U = 1.5
D = 0.1
H = 4.1*D
# inlet = fe.Expression(("4*1.5*sin(pi*t/8)*x[1]*(H-x[1])/(H*H)","0"), U=U, H=H, t=0, degree=2)
inlet = fe.Expression(("4*1.5*x[1]*(H-x[1])/(H*H)","0"), U=U, H=H, degree=2)
zero_v = fe.Constant( (0,0) )
zero = fe.Constant(0)
u_bcs = {
        1: {'type': 'dirichlet', 'value': inlet},
        2: {'type': 'dirichlet', 'value': zero_v},
        4: {'type': 'dirichlet', 'value': zero_v},
        5: {'type': 'dirichlet', 'value': zero_v}
        }

p_bcs = {3: {'type': 'dirichlet', 'value': zero}}

c_bcs = {1: {'type': 'dirichlet', 'value': fe.Constant(1)}}

bc_dict = {
        'u': u_bcs,
        'p': p_bcs,
        'c': c_bcs
          }

coupled_physics.set_bcs(bc_dict)

# Solve this problem using a nonlinear solver
la_solver = fe.LUSolver()
solver = PhysicsSolver(coupled_physics, la_solver)

# Write solution
coupled_physics.set_writer("output", "pvd")

t = 0
while t < 5:
    t += dt
    solver.solve()
    coupled_physics.write(time_stamp=t)
    coupled_physics.update_previous_solution()



