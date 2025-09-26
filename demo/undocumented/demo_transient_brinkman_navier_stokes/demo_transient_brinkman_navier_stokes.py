import dolfinx
import numpy as np
import psutil as ps 
import os 
import ufl

from flatiron_tk.functions import build_field_scalar_function
from flatiron_tk.mesh import Mesh
from flatiron_tk.physics import TransientBrinkmanNavierStokes
from flatiron_tk.solver import BlockNonLinearSolver
from flatiron_tk.solver import BlockSplitTree
from flatiron_tk.solver  import ConvergenceMonitor
from flatiron_tk.solver import NonLinearProblem
from mpi4py import MPI
from petsc4py import PETSc

# Define Mesh and Fictitious Domain
mesh_file = '../mesh/2db.msh'
fic_dom_file = '../mesh/2dfd.msh'

mesh = Mesh(mesh_file=mesh_file)
fic_dom = Mesh(mesh_file=fic_dom_file)

def inlet_velocity(x):
    values = np.zeros((2, x.shape[1]), dtype=dolfinx.default_scalar_type)
    y = x[1]
    H = 1.0  # Height from -0.5 to 0.5
    y_shifted = y + 0.5  # Shift y so bounds are [0, 1]
    U_max = 4
    values[0] = 4 * U_max * y_shifted * (H - y_shifted) / (H ** 2)
    return values

def no_slip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def zero_scalar(x):
    return np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type)

# Define Problem
bnse = TransientBrinkmanNavierStokes(mesh)
bnse.set_element('CG', 1, 'CG', 1)
bnse.build_function_space()

# Get Velocity (VV [CG, 1, dim = 2]) and Pressure (VS [CG, 1, dim = 1]) function spaces
VV = bnse.get_function_space('u').collapse()[0]
VS = bnse.get_function_space('p').collapse()[0]

# Porous domain properties
indicator = build_field_scalar_function(mesh, fic_dom, 1.0, 0.0, 'I')
bnse.set_indicator_function(indicator)

# Permeability
permeability_value = 1e-4
permeability = dolfinx.fem.Function(VS)
permeability.x.array[:] = permeability_value
permeability.x.scatter_forward()
bnse.set_permeability(permeability)

# Fluid properties
dt = 0.05
mu = 0.001
rho = 1

# Set problem parameters
bnse.set_time_step_size(dt)
bnse.set_midpoint_theta(0.5)
bnse.set_density(rho)
bnse.set_dynamic_viscosity(mu)
bnse.set_weak_form(stab=True)

# Create boundary conditions
inlet_v = dolfinx.fem.Function(VV)
inlet_v.interpolate(lambda x: inlet_velocity(x))

zero_vector = dolfinx.fem.Function(VV)
zero_vector.interpolate(no_slip)

zero_p = dolfinx.fem.Function(VS)
zero_p.interpolate(zero_scalar)

u_bcs = {
        10: {'type': 'dirichlet', 'value': inlet_v},
        13: {'type': 'dirichlet', 'value': zero_vector}
    }

p_bcs = {
        11: {'type': 'dirichlet', 'value': zero_p},
        12: {'type': 'dirichlet', 'value': zero_p}
    }
bc_dict = {'u': u_bcs, 
           'p': p_bcs}

bnse.set_bcs(bc_dict)
bnse.set_writer('output', 'pvd')

# Set problem 
problem = NonLinearProblem(bnse)

# Define block solver parameters 
def set_ksp_u(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setMonitor(ConvergenceMonitor("|----KSPU", verbose=True))
    ksp.setTolerances(max_it=3)
    ksp.pc.setType(PETSc.PC.Type.JACOBI)
    ksp.setUp()

def set_ksp_p(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setMonitor(ConvergenceMonitor("|--------KSPP", verbose=True))
    ksp.setTolerances(max_it=5)
    ksp.pc.setType(PETSc.PC.Type.HYPRE)
    ksp.pc.setHYPREType("boomeramg")
    ksp.setUp()

def set_outer_ksp(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setGMRESRestart(30)
    ksp.setTolerances(rtol=1e-10, atol=1e-10)
    ksp.setMonitor(ConvergenceMonitor("Outer ksp"))

split = {'fields': ('u', 'p'),
            'composite_type': 'schur',
            'schur_fact_type': 'full',
            'schur_pre_type': 'a11',
            'ksp0_set_function': set_ksp_u,
            'ksp1_set_function': set_ksp_p}

tree = BlockSplitTree(bnse, splits=split)
solver = BlockNonLinearSolver(tree, MPI.COMM_WORLD, problem, outer_ksp_set_function=set_outer_ksp)
solver.solve()

t = 0.0
while t < 5.0:
    print(f'Solving at time t = {t:.2f}')
    
    # # Set the inlet velocity for the current time step
    # inlet_v.interpolate(lambda x: inlet_velocity(x))
    
    # Solve the problem
    solver.solve()

    bnse.update_previous_solution()
    bnse.write(time_stamp=t)
    t += dt

    process = ps.Process(os.getpid())
    mem_info = process.memory_info()
    rss_memory_MB = mem_info.rss / (1024 * 1024)  # Convert bytes to MB
    print(f"Current RSS memory usage: {rss_memory_MB:.2f} MB")

# Compute flow rates 
n = mesh.get_facet_normal()
u = bnse.solution.sub(0).collapse()

form_flow_rate_in = dolfinx.fem.form(ufl.inner(u, n) * bnse.ds(10))
form_flow_rate_out_1 = dolfinx.fem.form(ufl.inner(u, n) * bnse.ds(11))
form_flow_rate_out_2 = dolfinx.fem.form(ufl.inner(u, n) * bnse.ds(12))
flow_rate_in = np.abs(dolfinx.fem.assemble_scalar(form_flow_rate_in))
flow_rate_out_1 = np.abs(dolfinx.fem.assemble_scalar(form_flow_rate_out_1))
flow_rate_out_2 = np.abs(dolfinx.fem.assemble_scalar(form_flow_rate_out_2))

form_inlet_pressure = dolfinx.fem.form(bnse.solution.sub(1).collapse() * bnse.ds(10))
form_outlet_pressure_1 = dolfinx.fem.form(bnse.solution.sub(1).collapse() * bnse.ds(11))
form_outlet_pressure_2 = dolfinx.fem.form(bnse.solution.sub(1).collapse() * bnse.ds(12))
inlet_pressure = dolfinx.fem.assemble_scalar(form_inlet_pressure)
outlet_pressure = dolfinx.fem.assemble_scalar(form_outlet_pressure_1)
delta_p = inlet_pressure - outlet_pressure

total_resistance_fem = delta_p / flow_rate_in


print(f'Inlet flow rate: {flow_rate_in:.6f}')
print(f'Outlet flow rate 1: {flow_rate_out_1:.6f}')
print(f'Outlet flow rate 2: {flow_rate_out_2:.6f}')
print(f'Ratio out1/out2: {flow_rate_out_1/flow_rate_out_2:.6f}')
print(f'Mass conservation check (in - out1 - out2): {flow_rate_in - flow_rate_out_1 - flow_rate_out_2:.6e}')


resistance_kc = mu / permeability_value
print(f'Resistance from Brinkman term: {resistance_kc:.6f}')

resistance_pipe = (12 * mu) 
print(f'Resistance from Poiseuille term: {resistance_pipe:.6f}')

total_resistance_analytical = resistance_pipe + resistance_kc
print(f'Total analytical resistance: {total_resistance_analytical:.6f}')

analytical_resistance = 12 * mu + mu / permeability_value

print(f'Analytical resistance: {analytical_resistance:.6f}')

print('\n'*2)
print('Comparison of FEM and analytical resistance:')
print(f'FEM resistance: {total_resistance_fem:.6f}')
print(f'Analytical resistance: {analytical_resistance:.6f}')
print(f'Relative error: {np.abs(total_resistance_fem - analytical_resistance) / analytical_resistance * 100:.2f}%')