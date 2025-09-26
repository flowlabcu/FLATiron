import dolfinx
import numpy as np

from flatiron_tk.mesh import Mesh
from flatiron_tk.physics import TransientNavierStokes
from flatiron_tk.solver  import ConvergenceMonitor
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver

# Define boundary condition functions
def no_slip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def zero_pressure(x):
    return np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type)

# Define the mesh
mesh_file = '../mesh/foc.msh'
mesh = Mesh(mesh_file=mesh_file)

# Create transient Navier-Stokes object
nse = TransientNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()

# Physical parameters
dt = 0.05
mu = 0.001
rho = 1
u_mag = 4 

nse.set_time_step_size(dt)
nse.set_midpoint_theta(0.5)
nse.set_density(rho)
nse.set_dynamic_viscosity(mu)
nse.set_weak_form(stab=True)

# Get function spaces for boundary conditions functions
V_u = nse.get_function_space('u').collapse()[0]
V_p = nse.get_function_space('p').collapse()[0]

# Parabolic profile 
def inlet_velocity(x):
    # Parabolic profile: u_x = 4 * U_max * y * (H - y) / H^2
    # Assuming inlet along x, y in [0, H], U_max = 10.0, H = 4.1
    values = np.zeros((2, x.shape[1]), dtype=dolfinx.default_scalar_type)
    y = x[1]
    H = 4.1
    U_max = u_mag
    values[0] = 4 * U_max * y * (H - y) / (H ** 2)
    return values

inlet_v = dolfinx.fem.Function(V_u)
inlet_v.interpolate(lambda x: inlet_velocity(x))

zero_p = dolfinx.fem.Function(V_p)
zero_p.interpolate(zero_pressure)

zero_v = dolfinx.fem.Function(V_u)
zero_v.interpolate(no_slip)

u_bcs = {
        1: {'type': 'dirichlet', 'value': inlet_v},
        2: {'type': 'dirichlet', 'value': zero_v},
        4: {'type': 'dirichlet', 'value': zero_v},
        5: {'type': 'dirichlet', 'value': zero_v}
        }

p_bcs = {
        3: {'type': 'dirichlet', 'value': zero_p},
        }

bc_dict = {'u': u_bcs, 
           'p': p_bcs}

nse.set_bcs(bc_dict)

# Set the output writer
nse.set_writer('output', 'pvd')

# Set the problem 
problem = NonLinearProblem(nse)

# Set the solver
def my_custom_ksp_setup(ksp):
    ksp.setType(ksp.Type.FGMRES)        
    ksp.pc.setType(ksp.pc.Type.LU)  
    ksp.setTolerances(rtol=1e-12, atol=1e-10, max_it=500)
    ksp.setMonitor(ConvergenceMonitor('ksp'))

solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)

# Solve
while t < 10.0:
    print(f'Solving at time t = {t:.2f}')
    
    # Set the inlet velocity for the current time step
    inlet_v.interpolate(lambda x: inlet_velocity(x))
    
    # Solve the problem
    solver.solve()

    nse.update_previous_solution()
    nse.write(time_stamp=t)
    
    # Update time
    t += dt
