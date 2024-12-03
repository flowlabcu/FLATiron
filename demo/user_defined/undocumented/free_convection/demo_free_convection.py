import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

# ------------------------------------------------------- #

import fenics as fe
from flatiron_tk.mesh import Mesh
from flatiron_tk.physics import IncompressibleNavierStokes, TransientMultiPhysicsProblem, TransientScalarTransport
from flatiron_tk.solver import PhysicsSolver


# Define fluid parameters
dt = 0.01
mu = 1.8e-5
rho = 1.3e-3

# Define thermal parameters
Pr = 1
Ra = 1e5
delta_temp = 1.0
length = 1.0
gravity = -9.81
specific_heat = 1.0
alpha = mu/Pr/rho
conductivity = alpha * rho * specific_heat
print(alpha)

# Define thermal expansion coefficient
expansion_coef = (mu * conductivity * Ra) / (rho * delta_temp * gravity * length**3)

# Read in mesh file
mesh_file = 'unit_square.h5'
mesh = Mesh(mesh_file=mesh_file)

# Set nse
nse = IncompressibleNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.set_time_step_size(dt)
nse.set_mid_point_theta(0.5)
nse.set_density(rho)
nse.set_dynamic_viscosity(mu)

# Set stp
stp = TransientScalarTransport(mesh, dt, theta=0.5)
stp.tag = 'T'
stp.set_element('CG', 1)
stp.set_diffusivity(alpha, alpha)
stp.set_reaction(0.0, 0.0)

# Set coupled physics
coupled_physics = TransientMultiPhysicsProblem(nse, stp)
coupled_physics.set_element()
coupled_physics.build_function_space()
u = coupled_physics.solution_function('u')
p = coupled_physics.solution_function('p')
T = coupled_physics.solution_function('T')

# Get solution values for coupling term
u0 = fe.split(coupled_physics.sub_physics[0].previous_solution)[0]
T0 = coupled_physics.sub_physics[1].previous_solution
stp.set_advection_velocity(u0, u)

# Add stabilization options for both physics classes
nse_options = {'stab': True}
stp_options = {'stab': False}
coupled_physics.set_weak_form(nse_options, stp_options)

# Define g as the gravity vector
g = fe.Constant((0, gravity))

# Get the test function for w for adding coupling term
w = coupled_physics.test_function('u')

# Define boussinesq approximation body force for 'compressibility'
boussinesq_body_force = 0.5*((1 - T)*expansion_coef*g + (1 - T0)*expansion_coef*g)
coupled_physics.add_to_weakform(fe.dot(w, boussinesq_body_force), stp.dx)

# Assign boundary conditions
zero_vector = fe.Constant((0,0))
u_bcs = {
    1: {'type': 'dirichlet', 'value': zero_vector},
    2: {'type': 'dirichlet', 'value': zero_vector},
    3: {'type': 'dirichlet', 'value': zero_vector},
    4: {'type': 'dirichlet', 'value': zero_vector}
}

T_bcs = {
    1: {'type': 'dirichlet', 'value': fe.Constant(-0.5*delta_temp)},
    3: {'type': 'dirichlet', 'value': fe.Constant(0.5*delta_temp)}
}

# Point pressure boundary condition in lower left corner
p_bcs = {'point_0': {'type': 'dirichlet', 'value':'zero', 'x': (0., 0.)}}

# Combine boundary conditions and send to coupled physics
bc_dict = {
    'u': u_bcs,
    'p': p_bcs,
    'T': T_bcs
}

coupled_physics.set_bcs(bc_dict)

# Solve this problem using a nonlinear solver
la_solver = fe.LUSolver()
solver = PhysicsSolver(coupled_physics, la_solver)

# Write solution
coupled_physics.set_writer("output", "pvd")

# Time loop
t = 0
while t < 1.0:
    print('-'*50 + '\n solved time = {}\n'.format(t) + '-'*50)
    solver.solve()
    coupled_physics.write(time_stamp=t)
    coupled_physics.update_previous_solution()
    t += dt

