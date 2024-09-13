'''
Demo for 1D advection-diffusion-reaction equation [0,1]
This problem recreates fig 2.1 in Donea's book: Finite Element Methods for Flow Problems

u*dc/dx - D*d^2c/dx^2 = 1
c[0] = 0
c[1] = 0

c = 1/u * ( x - (1-exp(g*x))/(1-exp(g)) )
g = u/D

This demo demonstrate stabilized adr solver using the scalar transport class
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

import fenics as fe
from flatiron_tk.physics import ScalarTransport
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import PhysicsSolver

# Define mesh
ne = 10
IM = fe.IntervalMesh(ne, 0, 1)
h = 1/ne
mesh = Mesh(mesh=IM)

# Mark mesh
def left(x, left_bnd):
    return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
def right(x, right_bnd):
    return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
mesh.mark_boundary(1, left, (0.))
mesh.mark_boundary(2, right, (1.))

# Define problem
st = ScalarTransport(mesh)
st.set_element('CG', 1)
st.build_function_space()

# Set constants
u = 1
Pe = 5
D = u/Pe/2*h
r = 1.
st.set_advection_velocity(u)
st.set_diffusivity(D)
st.set_reaction(r)

# Set weak form
st.set_weak_form()

# Add supg term
st.add_stab('su')

# Set bc
bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.)},
           2:{'type': 'dirichlet', 'value': fe.Constant(0.)}}
st.set_bcs(bc_dict)

# Set solver
la_solver = fe.PETScKrylovSolver()
fe.PETScOptions.set("ksp_monitor")
la_solver.set_from_options()
solver = PhysicsSolver(st, la_solver=la_solver)

# Solve
solver.solve()
st.set_writer('output', 'h5')
st.write()

# Plot solution
x = np.linspace(0, 1, 100*(ne+1))
g = u/D
sol_exact = 1/u * (x - (1-np.exp(g*x))/(1-np.exp(g)))
fe.plot(st.solution, linestyle='-', marker='o', label='Computed solution')
plt.plot(x, sol_exact, 'r--', label='Exact solution')
plt.grid(True)
plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel('x')
plt.ylabel('c')
plt.legend()
plt.savefig('demo_steady_adr.png')
plt.show()
