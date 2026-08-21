import dolfinx
import flatiron_tk
import numpy as np
import ufl

from flatiron_tk.mesh import RectMesh, Boundary
from flatiron_tk.physics import TransientNavierStokes
from flatiron_tk.physics import TransientScalarTransport
from flatiron_tk.physics import TransientMultiPhysicsProblem
from flatiron_tk.solver import ConvergenceMonitor
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver
"""
This verification problem models the boussinesq approximation problem for free convection as 
described in Donea et al. (2003) "Finite Element Methods for Flow Problems" Chapter 6.8.4.3.
"""

# Donea modifies the Rayleigh nummber and holds the Prandtl number
# as a constant Pr=1

# Higher Ra drives stronger buoyancy forcing and thinner thermal/momentum
# boundary layers (delta ~ Ra^-1/4), so the mesh must be refined and the time
# step reduced as Ra grows to keep the simulation stable and resolved.
TOTAL_TIME = 1.0

RA_SIM_PARAMS = {
    1e3: {'h': 1/64,  'dt': 1e-3, 'num_steps': int(TOTAL_TIME/1e-3)},
    1e4: {'h': 1/96,  'dt': 5e-4, 'num_steps': int(TOTAL_TIME/5e-4)},
    1e5: {'h': 1/128, 'dt': 1e-4, 'num_steps': int(TOTAL_TIME/1e-4)},
    1e6: {'h': 1/192, 'dt': 2e-5, 'num_steps': int(TOTAL_TIME/2e-5)},
}

# DEFINE ALL FLUID AND THERMAL PROPERTIES
PR = 1.0
ALPHA = 1.0
RHO = 1.0
MU = 1.0
GRAVITY = -1.0
DELTA_TEMP = 1.0
LC = 1.0

def _compute_expansion_coefficient(Ra):
    """Compute the expansion coefficient based on the Rayleigh number."""
    nu = MU / RHO  # Kinematic viscosity
    return Ra * nu * ALPHA / (GRAVITY * LC**3 * DELTA_TEMP)

def run_simulation(Ra, output_dir=None):
    if output_dir is None:
        exp = int(round(np.log10(Ra)))
        output_dir = f'output_Ra1e{exp}'
    sim_params = RA_SIM_PARAMS[Ra]
    DT = sim_params['dt']
    num_steps = sim_params['num_steps']
    # Generate mesh
    mesh = RectMesh(0, 0, LC, LC, sim_params['h'])
    Left = Boundary(mesh, 1)
    Bottom = Boundary(mesh, 2)
    Right = Boundary(mesh, 3)
    Top = Boundary(mesh, 4)

    nse = TransientNavierStokes(mesh)
    nse.set_element('CG', 1, 'CG', 1)
    nse.build_function_space()
    nse.set_time_step_size(DT)
    nse.set_midpoint_theta(0.5)
    nse.set_density(RHO)
    nse.set_dynamic_viscosity(MU)

    adr = TransientScalarTransport(mesh)
    adr.set_tag('T')
    adr.set_element('CG', 1)
    adr.set_time_step_size(DT)
    adr.set_diffusivity(ALPHA, ALPHA)
    adr.set_reaction(0.0, 0.0)

    coupled_physics = TransientMultiPhysicsProblem(nse, adr)
    coupled_physics.set_element()
    coupled_physics.build_function_space()

    # Get functions and test/trial functions for later
    p = nse.get_solution_function('p')
    q = nse.get_test_function('p')
    u = nse.get_solution_function('u')
    T = adr.get_solution_function('T')
    T0 = ufl.split(coupled_physics.sub_physics[1].previous_solution)[0]
    w = coupled_physics.sub_physics[0].get_test_function('u') 

    # Set weak forms
    adr.set_advection_velocity(u, u)
    nse_options = {'stab': True}
    adr_options = {'stab': True}
    coupled_physics.set_weak_form(nse_options, adr_options)

    beta = _compute_expansion_coefficient(Ra)
    g = ufl.as_vector([0.0, GRAVITY])
    boussinesq_term = 0.5 * ((1 - T) * beta * g + (1 - T0) * beta * g)

    coupled_physics.add_to_weak_form(ufl.inner(boussinesq_term, w) * nse.dx)

    V_u = nse.get_function_space('u').collapse()[0]

    no_slip = dolfinx.fem.Function(V_u); no_slip.x.array[:] = 0.0


    u_bcs = {Top.id: {'type': 'dirichlet', 'value': no_slip},
             Bottom.id: {'type': 'dirichlet', 'value': no_slip},
             Right.id: {'type': 'dirichlet', 'value': no_slip},
             Left.id: {'type': 'dirichlet', 'value': no_slip}}

    p_bcs = {}

    p_ref = flatiron_tk.constant(mesh, 0.0)
    eps = 1e-10
    pressure_penalty = eps * ufl.inner(p - p_ref, q) * nse.dx
    coupled_physics.add_to_weak_form(pressure_penalty)

    T_bcs = {Left.id: {'type': 'dirichlet', 'value': flatiron_tk.constant(mesh, -0.5*DELTA_TEMP)},
             Right.id: {'type': 'dirichlet', 'value': flatiron_tk.constant(mesh, 0.5*DELTA_TEMP)},}

    bc_dict = {
        'u': u_bcs,
        'p': p_bcs,
        'T': T_bcs
    }

    coupled_physics.set_bcs(bc_dict)
    coupled_physics.set_writer(f'output-Ra{Ra:.0e}', 'pvd')

    problem = NonLinearProblem(coupled_physics)

    def my_custom_ksp_setup(ksp):
        ksp.setType(ksp.Type.FGMRES)        
        ksp.pc.setType(ksp.pc.Type.LU)  
        ksp.setTolerances(rtol=1e-6, atol=1e-8, max_it=500)
        ksp.setMonitor(ConvergenceMonitor('ksp', verbose=False))

    solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)

    for step in range(num_steps):
        print(f'Solving time step {step+1}/{num_steps}, Time: {(step+1)*DT:.4f}')
        solver.solve()
        coupled_physics.update_previous_solution()

    coupled_physics.write()

    return 

def main():
    Rayleigh_numbers = [1e3, 1e4, 1e5, 1e6]
    for Ra in Rayleigh_numbers:
        print(f"Running simulation for Rayleigh number: {Ra}")
        run_simulation(Ra)

if __name__ == '__main__':
    main()
