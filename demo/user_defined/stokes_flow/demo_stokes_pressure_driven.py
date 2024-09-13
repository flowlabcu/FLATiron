import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------- #

import fenics as fe
from flatiron_tk.physics import StokesFlow
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import PhysicsSolver
import sys

# Define mesh
lx = 1
ly = 0.1
ney = 10
nex = int(ney * (lx//ly))
RM = fe.RectangleMesh(fe.Point(0, 0), fe.Point(lx, ly), nex, ney, 'crossed')
mesh = Mesh(mesh=RM)

# Mark mesh
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
mesh.mark_boundary(3, right, (lx))
mesh.mark_boundary(4, top, (ly))

# Define problem
ics = StokesFlow(mesh)
ics.set_element('CG', 1, 'CG', 1)
ics.build_function_space()

# Set coefficients on each term
# here since we are in transient mode, we have to set
# the function defining the previous and current time step.
# Since D and u are constants, we repeat the values for both entry
ics.set_body_force(fe.Constant((0,0)))
nu = 1.
ics.set_kinematic_viscosity(nu)

# Set weak form
ics.set_weak_form()

# Set stabilization
ics.add_stab()

# Set bc
u_bcs = {
        2: {'type': 'dirichlet', 'value': 'zero'},
        4: {'type': 'dirichlet', 'value': 'zero'},
        }
p_bcs = {1: {'type': 'neumann', 'value': fe.Constant(1)},
         3: {'type': 'dirichlet', 'value': 'zero'}}
bc_dict = {'u': u_bcs, 
           'p': p_bcs}
ics.set_bcs(bc_dict)

# Setup io
ics.set_writer('output', 'h5')

# Set problem
la_solver = fe.LUSolver()
solver = PhysicsSolver(ics, la_solver)

# Solve
solver.solve()

# Plot diagnostics
u, p = ics.solution_function().split(deepcopy=True)
x = np.linspace(0, lx, nex+1)
plt.figure()
fe.plot(u)
plt.title('Velocity field')
pc = []
for xi in x:
    pc.append(p(fe.Point(xi, ly/2)))
pc = np.array(pc)
plt.figure()
plt.plot(x, pc)
plt.title('Centerline pressure')
plt.show()


