'''
Demo for 1D advection-diffusion-reaction equation [0,1]
This problem recreates fig 2.1 in Donea's book: Finite Element Methods for Flow Problems

u*dc/dx - D*d^2c/dx^2 = 1
c[0] = 0
c[1] = 0

c = 1/u * ( x - (1-exp(g*x))/(1-exp(g)) )
g = u/D

This demo demonstrates stabilized adr solver using the scalar transport class
'''

import numpy as np
import matplotlib.pyplot as plt
from flatiron_tk.physics import ScalarTransport
from flatiron_tk.mesh import Mesh, LineMesh
from flatiron_tk.solver import PhysicsSolver

# Define mesh
h = 0.1
mesh = LineMesh(0, 1, h)

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
bc_dict = {1:{'type': 'dirichlet', 'value': 0.},
           2:{'type': 'dirichlet', 'value': 0.}}
st.set_bcs(bc_dict)

# Set solver
solver = PhysicsSolver(st)

# Solve
solver.solve()
st.set_writer('output', 'h5')
st.write()

# Plot solution
x = np.linspace(0, 1, int(1/h)+1)
g = u/D
sol_exact = 1/u * (x - (1-np.exp(g*x))/(1-np.exp(g)))
import fenics as fe
# Here we will plot with FEniCS
fe.plot(st.solution_function(), linestyle='-', marker='o', label='Computed solution')
plt.plot(x, sol_exact, 'r--', label='Exact solution')
plt.grid(True)
plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel('x')
plt.ylabel('c')
plt.legend()
plt.savefig('demo_steady_adr.png')
