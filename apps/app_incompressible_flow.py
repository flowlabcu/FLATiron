import dolfinx
import flatiron_tk
import numpy as np  
import sys
import ufl

from common import *
from flatiron_tk.functions import PlugInletProfile
from flatiron_tk.io import InputObject
from flatiron_tk.mesh import Mesh
from flatiron_tk.physics import SteadyStokes
from flatiron_tk.physics import SteadyNavierStokes
from flatiron_tk.physics import TransientNavierStokes
from mpi4py import MPI

def _parse_bc_input(input_object, physics):
    bcs_input = []

    u_bc_inputs = []
    p_bc_inputs = []
    for key in input_object.input_dict.keys():
        if not key.startswith('BC'):
            continue

        # bc_input will be (bnd_id, bc_type, options....)
        bc_input = input_object(key)
        bc_input = list(bc_input)
        try:
            bc_input[0] = int(bc_input[0])
        except ValueError:
            pass
        bc_type = bc_input[1]

        if bc_type in ['inlet', 'wall']:
            u_bc_inputs.append(bc_input)
        elif bc_type in ['pressure']:
            p_bc_inputs.append(bc_input)

    return u_bc_inputs, p_bc_inputs

def _build_u_bcs(input_object, physics, u_bc_inputs):

    avail_bc_type = ['inlet', 'wall']

    u_bc_dict = {}
    for bc_input in u_bc_inputs:

        bnd_id = bc_input[0]
        bc_type = bc_input[1]
        bc_options = bc_input[2:]

        if bc_type == 'inlet':
            u_bc_dict[bnd_id] = _build_inlet_bc(bnd_id, input_object, physics, bc_options)

        elif bc_type == 'wall':
            u_bc_dict[bnd_id] = _build_wall_bc(bnd_id, input_object, physics, bc_options)

        else:
            emsg = custom_err_msg(bc_type, 'velocity boundary conditions', avail_bc_type)
            raise ValueError(emsg)
    
    for k, v in u_bc_dict.items():
        if physics.mesh.comm.rank == 0:
            print(f'Boundary {k}: {v["type"]} with value {v["value"].name}')
    return u_bc_dict

def _build_inlet_bc(bnd_id, input_object, physics, bc_options):
    """"""
    profile_type = bc_options[0]
    flow_direction = input_object(bc_options[1])

    print(f'Building inlet BC with {profile_type} profile in direction {flow_direction}')

    if flow_direction == 'wall':
        flow_direction = -physics.mesh.get_mean_boundary_normal(bnd_id)

    centerline_speed = input_object(bc_options[2])
    avail_profile_type = ['plug', 'parabolic']

    # Evaluate profile
    inlet_velocity = dolfinx.fem.Function(physics.get_function_space('u').collapse()[0])
    
    if profile_type == 'plug':
        print(f'Centerline speed: {centerline_speed}')
        profile = PlugInletProfile(speed=centerline_speed, normal=flow_direction)
        inlet_velocity.interpolate(profile)
        inlet_velocity.name = 'plug'
    
    elif profile_type == 'parabolic':
        flow_speed = input_object(bc_options[2])
        center = input_object(bc_options[3])
        face_radius = input_object(bc_options[4])

        print(f'Face radius: {face_radius}, center: {center}, flow speed: {flow_speed}')
        
        if physics.mesh.get_tdim() == 2:
            flow_rate = 4 * flow_speed * face_radius / 3
            profile = flatiron_tk.ParabolicInletProfile(flow_rate, face_radius, center, flow_direction)
            inlet_velocity.interpolate(profile)
            inlet_velocity.name = 'parabolic'
        
        elif physics.mesh.get_tdim() == 3:
            flow_rate = np.pi * face_radius**2 * flow_speed / 2
            profile = flatiron_tk.ParabolicInletProfile(flow_rate, face_radius, center, flow_direction)
            inlet_velocity.interpolate(profile)
            inlet_velocity.name = 'paraboloid'
            
    else:
        emsg = custom_err_msg(profile_type, 'inlet profile type', avail_profile_type)
        raise ValueError(emsg)
    
    
    return {'type': 'dirichlet', 'value': inlet_velocity}

def _build_wall_bc(bnd_id, input_object, physics, bc_options):
    """"""
    profile_type = bc_options[0]
    avail_profile_type = ['no slip', 'traction']

    if profile_type == 'no slip':
        zero_v = dolfinx.fem.Function(physics.get_function_space('u').collapse()[0])
        zero_v.x.array[:] = 0.0
        zero_v.name = 'no_slip'
        return {'type': 'dirichlet', 'value': zero_v}
    
    elif profile_type == 'traction':
        traction_value = input_object(bc_options[1])
        traction_value = flatiron_tk.constant(physics.mesh, traction_value)
        return {'type': 'neumann', 'value': traction_value}
    
    else:
        emsg = custom_err_msg(profile_type, 'wall profile type', avail_profile_type)
        raise ValueError(emsg)

def _build_p_bcs(input_object, physics, p_bc_inputs):
    """"""

    p_bc_dict = {}

    for bc_input in p_bc_inputs:
        
        bnd_id = bc_input[0]
        bc_type = bc_input[1]
        bc_options = bc_input[2:]

        if bc_type == 'pressure':
            value = _build_pressure_bc(bnd_id, input_object, physics, bc_options)
            if value is not None:
                p_bc_dict[bnd_id] = _build_pressure_bc(bnd_id, input_object, physics, bc_options)

        else:
            emsg = custom_err_msg(bc_type, 'pressure boundary conditions', ['pressure'])
            raise ValueError(emsg)
    
    return p_bc_dict

def _build_pressure_bc(bnd_id, input_object, physics, bc_options):
    """"""
    avail_location_type = ['face', 'face dirichlet', 'fix nullspace']
    location_type = bc_options[0]
    pressure_value = input_object(bc_options[1])

    if location_type == 'face':
        return {'type': 'neumann', 'value': flatiron_tk.constant(physics.mesh, pressure_value)}
    
    elif location_type == 'face dirichlet':
        p_dirichlet = dolfinx.fem.Function(physics.get_function_space('p').collapse()[0])
        p_dirichlet.x.array[:] = pressure_value
        p_dirichlet.name = 'p_dirichlet'
        return {'type': 'dirichlet', 'value': p_dirichlet}
    
    elif location_type == 'fix nullspace':
        p = physics.get_solution_function('p')
        q = physics.get_test_function('p')
        p_ref = flatiron_tk.constant(physics.mesh, pressure_value)
        eps = 1e-10
        pressure_penalty = eps * ufl.inner(p - p_ref, q) * physics.dx
        physics.add_to_weak_form(pressure_penalty)
        return None

    else:
        emsg = custom_err_msg(location_type, 'pressure type', avail_location_type)
        raise ValueError(emsg)

def build_physics(input_object, mesh=None):
    """"""
    # Load mesh
    mesh_file = input_object('mesh file')
    if mesh is None: mesh = Mesh(mesh_file=mesh_file)
    else: mesh = mesh

    # Create physics object
    physics_type = input_object('flow physics type').lower()
    if physics_type == 'steady stokes'.lower():
        if mesh.comm.rank == 0:
            print('Building steady stokes physics')
        physics = SteadyStokes(mesh)
    elif physics_type == 'steady navier stokes'.lower():
        print('Building steady navier stokes physics')
        physics = SteadyNavierStokes(mesh)
    elif physics_type == 'transient navier stokes'.lower():
        print('Building transient navier stokes physics')
        dt = input_object('time step size')
        physics = TransientNavierStokes(mesh, dt, theta=0.5)
    else:
        emsg = custom_err_msg(physics_type, 'flow physics type', ['steady stokes', 'steady navier stokes', 'transient navier stokes'])
        raise ValueError(emsg)

    # Set element and function space 
    element_type = input_object('element type')
    if element_type == 'linear':
        physics.set_element('CG', 1, 'CG', 1)
    elif element_type == 'taylor hood':
        physics.set_element('CG', 2, 'CG', 1)
    else:
        emsg = custom_err_msg(element_type, 'element type', ['linear', 'taylor hood'])
        raise ValueError(emsg)
    physics.build_function_space()

    # Set other parameters
    rho = input_object('density')
    mu = input_object('dynamic viscosity')
    if physics_type == 'steady stokes':
        physics.set_kinematic_viscosity(mu/rho)
    
    else:
        physics.set_density(rho)
        physics.set_dynamic_viscosity(mu)

    # # Set time dependent parameters
    # if physics_type == 'navier stokes':
    #     dt = input_object('time step size')
    #     physics.set_time_step_size(dt)
    #     physics.set_mid_point_theta(0.5)

    # set weak form
    physics.set_weak_form()
    if element_type == 'linear':
        physics.add_stab()

    u_bc_inputs, p_bc_inputs = _parse_bc_input(input_object, physics)
    u_bc_dict = _build_u_bcs(input_object, physics, u_bc_inputs)
    p_bc_dict = _build_p_bcs(input_object, physics, p_bc_inputs)
    bc_dict = {'u': u_bc_dict,
               'p': p_bc_dict}
    
    physics.set_bcs(bc_dict)

    # Set writer 
    output_prefix = input_object('output prefix')
    output_type = input_object('output type')
    physics.set_writer(output_prefix, output_type)
    return physics

def _solve_stokes(input_object, solver, physics):
    """"""
    solver.solve()
    physics.write()

def _solve_steasdy_nse(input_object, solver, physics):
    """
    Here we will solve the stokes flow version of the problem first before moving on to the navier stokes case
    """

    sub_viscosity_enabled = input_object('enable sub viscosity')
    if sub_viscosity_enabled:
        mu_list = sorted(input_object('intermediate sub viscosity'), reverse=True)
        for i, mu in enumerate(mu_list):
            if physics.mesh.comm.rank == 0:
                print(f'Solving NSE with sub-viscosity {mu}, step {i+1} of {len(mu_list)}')
                print('-'*50)

            physics.external_function_dict['dynamic_viscosity'].value = mu
            solver.solve()
            
            u, p = physics.solution.split()
            physics.set_initial_guess(u, p)

    else:
        solver.solve()

def _solve_tnse(input_object, solver, physics):
    # Set time
    dt = input_object('time step size')
    time_span = input_object('time span')
    save_every = input_object('save every')
    current_time = 0
    time_step = 0

    # Save initial condition
    if physics.mesh.comm.rank == 0:
        print('-'*50)
        print('Current time %f is written' % current_time)
        print('-'*50)
    physics.write(time_stamp=current_time)

    # Solve
    while current_time < time_span:

        just_written = False
        early_exit = False

        # Update time step
        time_step += 1
        current_time += dt
        try:
            solver.solve()
        except: # any error at all at the solve stage will result in an early exit
            early_exit = True

        # Check for quick exit on any rank, all rank exit
        if physics.mesh.comm.allreduce(early_exit, MPI.LOR):
            print('-'*50)
            print('Current time %f'%current_time)
            flatiron_tk.custom_warning_message('Solve step exit eairly in an ERROR.')
            print('-'*50)
            break

        physics.update_previous_solution()

        # Write solution
        if time_step%save_every == 0:
            physics.write(time_stamp=current_time)
            if physics.mesh.comm.rank == 0:
                print('-'*50)
                print('Current time %f is written' % current_time)
                print('-'*50)
            physics.mesh.comm.Barrier()
            just_written = True

    # Write final time step
    if not just_written:
        if physics.mesh.comm.rank == 0:
            print('-'*50)
            print('Current time %f is written' % current_time)
            print('-'*50)
        physics.write(time_stamp=current_time)

def main(input_file):
    input_object = InputObject(input_file)

    # Create physics object
    physics = build_physics(input_object)
    solver = build_solver(physics, input_object)

    if input_object('flow physics type').lower() == 'steady stokes':
        _solve_stokes(input_object, solver, physics)
    
    elif input_object('flow physics type').lower() == 'steady navier stokes':
        _solve_steasdy_nse(input_object, solver, physics)

    elif input_object('flow physics type').lower() == 'transient navier stokes':
        _solve_tnse(input_object, solver, physics)

if __name__ == '__main__':
    input_file = sys.argv[1]
    main(input_file)
    