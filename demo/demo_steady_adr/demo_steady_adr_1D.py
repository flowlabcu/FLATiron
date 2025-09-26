import matplotlib.pyplot as plt
import numpy as np

from flatiron_tk import constant
from flatiron_tk.mesh import LineMesh
from flatiron_tk.physics import SteadyScalarTransport
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver

# Create Mesh
mesh = LineMesh(0, 1, 1/10)

# Define Problem 
stp = SteadyScalarTransport(mesh, tag='c')
stp.set_writer('output', 'xdmf')
stp.set_element('CG', 1)
stp.build_function_space()

# Set parameters
stp.set_advection_velocity(0.0)
stp.set_diffusivity(1.0)
stp.set_reaction(0.0)

# Set weak form and stabilization
stp.set_weak_form()
stp.add_stab()  

# Define Boundary Conditions
left_bc = constant(mesh, 1.0)
right_bc = constant(mesh, 0.5)

bc_dict = {
    1: {'type': 'dirichlet', 'value': left_bc},
    2: {'type': 'neumann', 'value': right_bc}
}

stp.set_bcs(bc_dict)

# Define and Solve Problem
problem = NonLinearProblem(stp)
solver = NonLinearSolver(mesh.msh.comm, problem)

solver.solve()
stp.write()

# Extract solution for plotting
x = stp.mesh.msh.geometry.x[:, 0]  # Assuming 1D mesh, extract x-coordinates
u = stp.solution.x.array           # Solution values as a NumPy array

# Sort points for plotting (since mesh nodes may be unordered)
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
u_sorted = u[sorted_indices]

# Plot
plt.plot(x_sorted, u_sorted, marker='o', label="Solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("flatiron_tk 1D Steady Scalar Transport Solution")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("steady_scalar_transport_solution.png", dpi=300)
plt.show()

