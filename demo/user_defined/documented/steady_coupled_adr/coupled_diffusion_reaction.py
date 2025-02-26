import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

'''
Demo for a coupled diffusion reaction problem

D_A \frac{d^2A}{dx^2} - k_v A B = 0 
D_B \frac{d^2B}{dx^2} - 2k_v A B = 0
D_C \frac{d^2C}{dx^2} + k_v A B = 0

x \in [0, L]

BC: 
A(x=0) = C0
B(x=0) = C0
C(x=0) = 0

A'(x=L) = -k_S A B  / D_A
B'(x=L) = -2K_S A B / D_B
C'(x=L) = K_s A B   / D_C


'''

# ---------------------------------------------------------- #

import fenics as fe
from flatiron_tk.physics import MultiPhysicsProblem, ScalarTransport
from flatiron_tk.mesh import Mesh, LineMesh
from flatiron_tk.solver import PhysicsSolver

# Define mesh
ne = 10
h = 1/ne
mesh = LineMesh(0, 1, h)

# Define constants
# diffusion coefficients
D_A = 1.0
D_B = 1.0
D_C = 1.0 
k_v = 1 # Volumetric reaction rate
k_s = 1 # Surface reaction rate
C0 = 1 # Left BC for species A and B
u = 0 # No advection

# Define the problem for species A
A_pde = ScalarTransport(mesh, tag='A')
A_pde.set_element('CG', 1)
A_pde.set_advection_velocity(u)
A_pde.set_diffusivity(D_A)

# Define the problem for species B
B_pde = ScalarTransport(mesh, tag='B')
B_pde.set_element('CG', 1)
B_pde.set_advection_velocity(u)
B_pde.set_diffusivity(D_B)

# Define the problem for species C
C_pde = ScalarTransport(mesh, tag='C')
C_pde.set_element('CG', 1)
C_pde.set_advection_velocity(u)
C_pde.set_diffusivity(D_C)

# Define a multiphysics problem as a combination of physics of
# species A, B, C 
coupled_physics = MultiPhysicsProblem(A_pde, B_pde, C_pde)
coupled_physics.set_element()
coupled_physics.build_function_space()

# Set the coupling part of the equations
A = coupled_physics.solution_function('A')
B = coupled_physics.solution_function('B')
C = coupled_physics.solution_function('C')
A_pde.set_reaction(-k_v*A*B)
B_pde.set_reaction(-2*k_v*A*B)
C_pde.set_reaction(k_v*A*B)

# Set weakform
coupled_physics.set_weak_form()

# Set BCs for specific physics
A_bcs = {
        1: {'type': 'dirichlet', 'value': fe.Constant(C0)},
        2: {'type': 'neumann', 'value': -k_s*A*B/D_A}
        }

B_bcs = {
        1: {'type': 'dirichlet', 'value': fe.Constant(C0)},
        2: {'type': 'neumann', 'value': -2*k_s*A*B/D_B}
        }

C_bcs = {
        1: {'type': 'dirichlet', 'value': fe.Constant(0)},
        2: {'type': 'neumann', 'value': k_s*A*B/D_C}
        }

bc_dict = {
        'A': A_bcs,
        'B': B_bcs,
        'C': C_bcs
          }
coupled_physics.set_bcs(bc_dict)

# Solve this problem using a nonlinear solver
solver = PhysicsSolver(coupled_physics)
solver.solve()

# Write solution
coupled_physics.set_writer("output", "pvd")
coupled_physics.write()

# Plot solution
solutions = coupled_physics.solution_function().split(deepcopy=True)
fe.plot(solutions[0], label='A')
fe.plot(solutions[1], label='B')
fe.plot(solutions[2], label='C')
plt.ylim([-0.1, 1.1])
plt.legend()
plt.savefig('coupled_diffusion_reaction.png')




