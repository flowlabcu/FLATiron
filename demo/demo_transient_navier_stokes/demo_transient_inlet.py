import numpy as np
import matplotlib.pyplot as plt
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver
import dolfinx
import ufl 
import numpy as np
from flatiron_tk.physics import TransientNavierStokes
from flatiron_tk.solver  import ConvergenceMonitor
from flatiron_tk.mesh import Mesh
import sys
import basix
import subprocess
import psutil as ps 
import os 

process = ps.Process(os.getpid())

def no_slip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def zero_pressure(x):
    return np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type)

mesh_file = '../mesh/foc.msh'
mesh = Mesh(mesh_file=mesh_file)


nse = TransientNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()

dt = 0.05
mu = 0.001
rho = 1
nse.set_time_step_size(dt)
nse.set_midpoint_theta(0.5)

nse.set_density(rho)
nse.set_dynamic_viscosity(mu)
nse.set_weak_form(stab=True)

V_u = nse.get_function_space('u').collapse()[0]
V_p = nse.get_function_space('p').collapse()[0]

def inlet_velocity(x, t):
    values = np.zeros((2, x.shape[1]), dtype=dolfinx.default_scalar_type)
    values[0] = np.sin(t)
    return values

inlet_v = dolfinx.fem.Function(V_u)

# At each time step:
t = 0.0
inlet_v.interpolate(lambda x: inlet_velocity(x, t))

zero_p = dolfinx.fem.Function(V_p)
zero_p.interpolate(zero_pressure)

zero_v = dolfinx.fem.Function(V_u)
zero_v.interpolate(no_slip)


u_bcs = {
        1: {'type': 'dirichlet', 'value': inlet_v},
        2: {'type': 'dirichlet', 'value': zero_v},
        4: {'type': 'dirichlet', 'value': zero_v},
        5: {'type': 'dirichlet', 'value': zero_v},
        }

p_bcs = {
        3: {'type': 'dirichlet', 'value': zero_p},
        }

bc_dict = {'u': u_bcs, 
           'p': p_bcs}

nse.set_bcs(bc_dict)

nse.set_writer('output', 'pvd')

problem = NonLinearProblem(nse)
def my_custom_ksp_setup(ksp):
    ksp.setType(ksp.Type.FGMRES)        
    ksp.pc.setType(ksp.pc.Type.LU)  
    ksp.setTolerances(rtol=1e-12, atol=1e-10, max_it=500)
    ksp.setMonitor(ConvergenceMonitor('ksp'))

solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)

with open('mem.log', 'w') as mem_log:
    mem_log.write(f"Initial memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB\n")


count = 0 
mem = []
while count < 500:
    rss_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    mem.append(rss_memory)
    with open('mem.log', 'a') as mem_log:
        mem_log.write(f"Memory usage at time {t:.2f}: {rss_memory:.2f} MB\n")

    print(f'Solving at time t = {t:.2f}')
    
    # Set the inlet velocity for the current time step
    inlet_v.interpolate(lambda x: inlet_velocity(x, t))
    

    # Solve the problem
    solver.solve()

    nse.update_previous_solution()
    nse.write(time_stamp=t)
    
    # Update time
    t += dt
    count += 1

plt.plot(np.arange(len(mem)), mem, label='Memory Usage (MB)')
plt.xlabel('Time Step')
plt.ylabel('Memory Usage (MB)')
plt.savefig('memory_usage.png')
