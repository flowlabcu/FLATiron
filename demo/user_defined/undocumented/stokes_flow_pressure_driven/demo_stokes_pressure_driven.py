import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------- #

import fenics as fe
from flatiron_tk.physics import StokesFlow
from flatiron_tk.mesh import Mesh, RectMesh
from flatiron_tk.solver import PhysicsSolver

# # Define mesh
lx = 1
ly = 0.1
h = 1/100
mesh = RectMesh(0, 0, lx, ly, h)


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
x = np.linspace(0, lx, 101)
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


