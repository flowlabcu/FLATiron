import numpy as np
import dolfinx
from flatiron_tk.mesh import RectMesh
from flatiron_tk.physics import SteadyNavierStokes
from flatiron_tk.physics import MasslessTracerTracker

from flatiron_tk.solver  import ConvergenceMonitor
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver

def solve_nse(mesh, Re):
    # Define boundary conditions functions
    def no_slip(x):
        return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

    def u_inlet(x):
        return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

    # Build Navier-Stokes problem
    nse = SteadyNavierStokes(mesh)
    nse.set_element('CG', 1, 'CG', 1)
    nse.build_function_space()

    # Set physical parameters
    mu = 1.0 / Re
    rho = 1.0
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)

    # Set weak form and stabilization
    nse.set_weak_form()
    nse.add_stab()

    # Velocity and pressure subspaces
    V_u = nse.get_function_space('u').collapse()[0]
    V_p = nse.get_function_space('p').collapse()[0]

    # Create boundary condition functions
    zero_v = dolfinx.fem.Function(V_u)
    zero_v.interpolate(no_slip)
    inlet_v = dolfinx.fem.Function(V_u)
    inlet_v.interpolate(u_inlet)
    zero_p = dolfinx.fem.Function(V_p)
    zero_p.x.array[:] = 0.0

    # Define boundary conditions
    u_bcs = {
            1: {'type': 'dirichlet', 'value': zero_v},
            2: {'type': 'dirichlet', 'value': zero_v},
            3: {'type': 'dirichlet', 'value': zero_v},
            4: {'type': 'dirichlet', 'value': inlet_v},
            }

    p_bcs = {
            1: {'type': 'dirichlet', 'value': zero_p},
            }

    bc_dict = {'u': u_bcs, 
            'p': p_bcs}

    nse.set_bcs(bc_dict)

    nse.set_writer('output', 'pvd')

    # Define problem
    problem = NonLinearProblem(nse)

    # Custom KSP setup function
    def my_custom_ksp_setup(ksp):
        ksp.setType(ksp.Type.FGMRES)        
        ksp.pc.setType(ksp.pc.Type.LU)  
        ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)
        ksp.setMonitor(ConvergenceMonitor('ksp'))

    # Create nonlinear solver
    solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)
    
    # Solve the problem
    solver.solve()
    nse.write()

    return nse

# Create mesh
ne = 64
mesh = RectMesh(0.0, 0.0, 1.0, 1.0, 1/ne)

nse = solve_nse(mesh=mesh, Re=100.0)

# Set up particle tracker
particle_tracker = MasslessTracerTracker(mesh, dt=0.01)
particle_tracker.set_writer('output/particles')
particle_tracker.set_particle_positions_from_boundary(boundary_id=4)

# Shifting particles down to get them in the domain better (for lid driven cavity mesh)
particle_tracker.particle_positions[:, 1] -= 0.1 

# Write initial particle positions
particle_tracker.write()



u = nse.get_solution_function().sub(0)
for i in range(1000):
    particle_tracker.update_particle_positions(current_velocity=u, method='euler')
    particle_tracker.write(time_stamp=i)