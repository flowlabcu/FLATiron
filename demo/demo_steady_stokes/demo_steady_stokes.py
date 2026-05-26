import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl 

from flatiron_tk.solver  import ConvergenceMonitor
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver
from flatiron_tk.mesh import RectMesh
from flatiron_tk.physics import SteadyStokes

# Define body force term 
def body_force(x):
    bx = (1 - 4 * x[1] + 12 * x[1]**2 - 8 * x[1]**3) + \
         (-2 + 24 * x[1] - 72 * x[1]**2 + 48 * x[1]**3) * x[0] + \
         (12 - 48 * x[1] + 72 * x[1]**2 - 48 * x[1]**3) * x[0]**2 + \
         (-24 + 48 * x[1]) * x[0]**3 + \
         (12 - 24 * x[1]) * x[0]**4

    by = (-12 * x[1]**2 + 24 * x[1]**3 - 12 * x[1]**4) + \
         (4 - 24 * x[1] + 48 * x[1]**2 - 48 * x[1]**3 + 24 * x[1]**4) * x[0] + \
         (-12 + 72 * x[1] - 72 * x[1]**2) * x[0]**2 + \
         (8 - 48 * x[1] + 48 * x[1]**2) * x[0]**3

    return ufl.as_vector([bx, by]) 
   
# Define the mesh 
ne = 64
h = 1/ne
mesh = RectMesh(0, 0, 1, 1, h)

# Define the Stokes problem
stk = SteadyStokes(mesh)
stk.set_element('CG', 1, 'CG', 1)
stk.build_function_space()

# Physical parameters 
nu = 1.0
stk.set_kinematic_viscosity(nu)

# Define the body force 
x = ufl.SpatialCoordinate(mesh.msh)
stk.set_body_force(body_force(x))

# Set weak form and stabilization
stk.set_weak_form()
stk.add_stab()

# Create functions for boundary conditions on the appropriate function spaces
V_u = stk.get_function_space('u').collapse()[0]
V_p = stk.get_function_space('p').collapse()[0]

# Zero functions (scalar and vector)
zero_v = dolfinx.fem.Function(V_u)
zero_v.x.array[:] = 0; zero_v.x.scatter_forward()
zero_p = dolfinx.fem.Function(V_p)
zero_p.x.array[:] = 0; zero_p.x.scatter_forward()

# Boundary conditions
u_bcs = {
    1: {'type': 'dirichlet', 'value': zero_v},
    2: {'type': 'dirichlet', 'value': zero_v},
    3: {'type': 'dirichlet', 'value': zero_v},
    4: {'type': 'dirichlet', 'value': zero_v}
    }
p_bcs = {1: {'type': 'dirichlet', 'value': zero_p}}
bc_dict = {'u': u_bcs, 'p': p_bcs}

stk.set_bcs(bc_dict)

# Set solver and solve
stk.set_writer('output', 'pvd')
problem = NonLinearProblem(stk)
solver = NonLinearSolver(mesh.msh.comm, problem)
solver.solve()
stk.write()

# Define exact solution for error computation
def u_exact_solution(x):
    u0e = x[0]**2 * (1 - x[0])**2 * (2 * x[1] - 6 * x[1]**2 + 4 * x[1]**3)
    u1e = -x[1]**2 * (1 - x[1])**2 * (2 * x[0] - 6 * x[0]**2 + 4 * x[0]**3)
    return ufl.as_vector([u0e, u1e])

u_exact = dolfinx.fem.Function(V_u)
expr = dolfinx.fem.Expression(u_exact_solution(x), V_u.element.interpolation_points())
u_exact.interpolate(expr)

# Get numerical solution
u = stk.get_solution_function().sub(0).collapse()

# Compute error
error_L2 = np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u - u_exact, u - u_exact) * ufl.dx)))
print(f"L2 error in velocity: {error_L2}")

# Save exact solution to file
with dolfinx.io.VTKFile(mesh.msh.comm, "output/u_exact.pvd", "w") as vtk:
    vtk.write_function(u_exact(x))


