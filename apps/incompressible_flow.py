import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

from feFlow.io import InputObject
from feFlow.info import info, warning, error
from feFlow.physics import StokesFlow
from feFlow.physics import IncompressibleNavierStokes
from feFlow.physics import SteadyIncompressibleNavierStokes
from feFlow.io import h5_mod
from feFlow.mesh import Mesh
from feFlow.solver import PhysicsSolver
from feFlow.functions.profiles import parabolic_2d, parabolic_3d, plug
import fenics as fe
from mpi4py import MPI
from common import build_la_solver, custom_err_msg

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

    # bc_input will be (bnd_id, bc_type, options...)
    # where bc_type can either be 'inlet' or 'wall'

    u_bc_dict = {}
    avail_ubcs = ['inlet', 'wall']

    for bc_input in u_bc_inputs:

        bnd_id = bc_input[0]
        bc_type = bc_input[1]
        bc_options = bc_input[2:]

        if bc_type == 'inlet':
            u_bc_dict[bnd_id] = _build_inlet_bc(bnd_id, input_object, physics, bc_options)

        elif bc_type == 'wall':
            u_bc_dict[bnd_id] = _build_wall_bc(bnd_id, input_object, physics, bc_options)

        else:
            emsg = custom_err_msg(bc_type, 'velocity boundary conditions', avail_ubcs)
            error(ValueError, emsg)
    return u_bc_dict

def _build_inlet_bc(bnd_id, input_object, physics, bc_options):

    '''
    inlet bc_options are
    (
     profile_type # 0
     *args
    )

    aside from profile_type,
    these inputs are all encoded as the variable name inside the input file, so
    we will have to do input_object(variable_name) to get the real value
    '''

    profile_type = bc_options[0]
    flow_direction = input_object(bc_options[1])
    if flow_direction == 'wall':
        flow_direction = physics.mesh.flat_boundary_normal(bnd_id)
    centerline_speed = input_object(bc_options[2])
    avails = ['plug', 'parabolic', 'custom']

    # Evaluate profile
    if profile_type == 'plug':
        # inputs: ('plug', flow_direction, flow_speed)

        # Parse flow speed and direction
        flow_speed = input_object(bc_options[2])

        # Normalize if it is not already
        if abs( np.linalg.norm(flow_direction) - 1 ) > 1e-8:
            wmsg = 'Supplied flow direction of %s is not a unit vector. feFlow will normalize this flow direction.' % str(flow_direction)
            warning(wmsg)
            flow_direction = flow_direction/np.linalg.norm(flow_direction)

        # Return plug profile
        bc_func = fe.Constant(np.array(flow_direction)*flow_speed)

    elif profile_type == 'parabolic':
        # inputs: ('parabolic', flow_direction, flow_speed, center, face_radius)

        # Parse flow speed and direction
        flow_speed = input_object(bc_options[2])
        center = input_object(bc_options[3])
        face_radius = input_object(bc_options[4])

        # Normalize if it is not already
        # I an redoing this here because the custom option doesn't use flow_direction
        if abs( np.linalg.norm(flow_direction) - 1 ) > 1e-8:
            wmsg = 'Supplied flow direction of %s is not a unit vector. feFlow will normalized this flow_direction.' % str(flow_direction)
            warning(wmsg)
            flow_direction = flow_direction/np.linalg.norm(flow_direction)

        if physics.dim == 2:
            nx = flow_direction[0]
            ny = flow_direction[1]
            tx = ny ; ty = -nx
            xc = center ; R = face_radius
            U = flow_speed
            eta_s = '(tx*x[0] - tx*xc + ty*x[1] - ty*yc) / R'
            peta = '(1+%s)*(1-%s)'% (eta_s, eta_s)
            ux = '(%s) ? 0.0 : U*%s*nx' % ('%s >= 1 || %s <= -1'%(eta_s, eta_s), peta)
            uy = '(%s) ? 0.0 : U*%s*ny' % ('%s >= 1 || %s <= -1'%(eta_s, eta_s), peta)
            bc_func = fe.Expression( (ux, uy), degree=2, tx=tx, ty=ty, xc=xc[0], yc=xc[1], nx=nx, ny=ny, U=U, R=R)

        elif physics.dim == 3:
            xc = center
            R = face_radius
            U = centerline_speed
            nx, ny, nz = flow_direction
            r = 'sqrt( pow(x[0]-%f, 2) + pow(x[1]-%f, 2) + pow(x[2]-%f, 2) )' % (xc[0], xc[1], xc[2])
            u_form = 'U*( ( pow(R,2) - pow(%s, 2) )/pow(R, 2) )' % r
            ux_str = '%s*%f' % (u_form, nx)
            uy_str = '%s*%f' % (u_form, ny)
            uz_str = '%s*%f' % (u_form, nz)
            ux = '(%s) ? 0.0 : %s' % ("%s >= R"%r, ux_str)
            uy = '(%s) ? 0.0 : %s' % ("%s >= R"%r, uy_str)
            uz = '(%s) ? 0.0 : %s' % ("%s >= R"%r, uz_str)
            bc_func = fe.Expression( (ux, uy, uz), degree=2, R=R, U=U, nx=nx, ny=ny, nz=nz)

        else:
            raise ValueError("physics dimension can only be 2 or 3. Current physics dim is %d"%physics.dim)

    elif profile_type == 'custom':
        emsg = '`custom` profile type is currently not implemented'
        error(ValueError, emsg)

    else:
        emsg = custom_err_msg(profile_type, 'inlet profiles', avails)
        error(ValueError, emsg)

    return {'type': 'dirichlet', 'value': bc_func}

def _build_wall_bc(bnd_id, input_object, physics, bc_options):

    '''
    wall bc_options are
    (
     profile_type # 0
     *args
    )

    aside from profile_type,
    these inputs are all encoded as the variable name inside the input file, so
    we will have to do input_object(variable_name) to get the real value
    '''

    profile_type = bc_options[0]

    # bc_input will be (bnd_id, bc_type, options...)

    avails = ['no slip', 'traction']

    if profile_type == 'no slip':
        zero_v = fe.Constant([0. for i in range(physics.dim)])
        return {'type': 'dirichlet', 'value': zero_v}

    elif profile_type == 'traction':
        trac = input_object(bc_options[2])
        return {'type': 'neumann', 'value': trac}

    else:
        emsg = custom_err_msg(profile_type, 'wall conditions', avails)
        error(ValueError, emsg)

def _build_p_bcs(input_object, physics, p_bc_inputs):

    # bc_input will be (bnd_id, bc_type, options...)

    p_bc_dict = {}

    for bc_input in p_bc_inputs:

        bnd_id = bc_input[0]
        bc_type = bc_input[1] # should be 'pressure'
        bc_options = bc_input[2:]

        p_bc_dict[bnd_id] = _build_pressure_bc(input_object, physics, bc_options)

    return p_bc_dict

def _build_pressure_bc(input_object, physics, bc_options):

    # bc_options will be (location_type, pressure_value, options...)

    avail_location_type = ['face', 'face dirichlet' ,'point']
    location_type = bc_options[0]
    pressure_value = input_object(bc_options[1])

    if location_type == 'face':
        # options are (location_type, pressure_value)
        return {'type': 'neumann', 'value': fe.Constant(pressure_value)}

    elif location_type == 'face dirichlet':
        return {'type': 'dirichlet', 'value': fe.Constant(pressure_value)}

    elif location_type == 'point':
        point_location = input_object(bc_options[2])
        # options are (location_type, pressure_value, point_location, (optional) eps)
        # where eps is the size of the search box. Any point within a distance of eps
        # of point_location will be considered for the pressure bc.
        # if eps is not provided, eps = 3e-16 (machine precision)
        if len(bc_options) == 4:
            eps = bc_options[3]
        else:
            eps = 3e-16
        return {'type': 'dirichlet', 'value': fe.Constant(pressure_value), 'x': point_location, 'eps': eps}

    else:
        emsg = custom_err_msg(bc_type, 'pressure boundary condition location type', avail_location_type)
        error(ValueError, emsg)

def build_physics(input_object, mesh=None):

    # Load mesh
    mesh_file = input_object('mesh file')
    if mesh is None:
        mesh = Mesh(mesh_file=mesh_file)
    else:
        mesh = mesh

    # Build physics
    physics_type = input_object('flow physics type').lower()
    if physics_type == 'stokes':
        physics = StokesFlow(mesh)
    elif physics_type == 'steady navier stokes':
        physics = SteadyIncompressibleNavierStokes(mesh)
    elif physics_type == 'navier stokes':
        physics = IncompressibleNavierStokes(mesh)
    else:
        emsg = custom_err_msg(physics_type, 'flow physics type', ['stokes', 'steady navier stokes', 'navier stokes'])
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
    if physics_type == 'stokes':
        physics.set_kinematic_viscosity(mu/rho)
    else:
        physics.set_density(rho)
        physics.set_dynamic_viscosity(mu)

    # Set time dependent parameters
    if physics_type == 'navier stokes':
        dt = input_object('time step size')
        physics.set_time_step_size(dt)
        physics.set_mid_point_theta(0.5)

    # set weak form
    physics.set_weak_form()
    if element_type == 'linear':
        physics.add_stab()

    # Set bcs
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

def _solve_stk(input_object, stk_physics_solver, stk_physics):
    stk_physics_solver.solve()
    stk_physics.write()

def _solve_snse(input_object, snse_physics_solver, snse_physics):

    '''
    Here we will solve the stokes flow version of the problem first before moving on to the navier stokes case
    '''


    # Build and solve the stokes flow version
    stk_input_object = copy.deepcopy(input_object)
    stk_input_object.input_dict['flow physics type'] = 'stokes'
    stk_physics = build_physics(stk_input_object, snse_physics.mesh)
    stk_physics.dirichlet_bcs = snse_physics.dirichlet_bcs
    stk_physics_solver = PhysicsSolver(stk_physics, snse_physics_solver.la_solver)
    stk_physics_solver.solve()
    (stk_u, stk_p) = stk_physics.solution_function().split(deepcopy=True)

    # Set initial guess of snse physics
    snse_physics.set_initial_guess(stk_u, stk_p)

    # Solve with sub viscosity
    sub_viscosity_on = input_object('enable sub viscosity')
    if sub_viscosity_on:
        mus = np.flip(np.sort(np.array(input_object('intermediate sub viscosity'))))
        for i, mu in enumerate(mus):
            if snse_physics.mesh.comm.rank == 0:
                print('*'*50)
                print('Solving sub viscosity step number %d/%d. mu = %f' % ((i+1), len(mus), mu))
                print('*'*50)
            snse_physics.external_function_dict['dynamic viscosity'].assign(mu)
            snse_physics_solver.solve()
        # Set initial guess for the next step
        (sub_u, sub_p) = snse_physics.solution_function().split(deepcopy=True)
        snse_physics.set_initial_guess(sub_u, sub_p)
    else:
        snse_physics_solver.solve()

    # Write result
    snse_physics.external_function_dict['dynamic viscosity'].assign(input_object('dynamic viscosity'))
    snse_physics_solver.solve()
    snse_physics.write()

def _solve_tnse(input_object, tnse_physics_solver, tnse_physics):

    # Set time
    dt = input_object('time step size')
    time_span = input_object('time span')
    save_every = input_object('save every')
    current_time = 0
    time_step = 0

    # TODO: Set initial condition


    # Save initial condition
    if tnse_physics.mesh.comm.rank == 0:
        print('-'*50)
        print('Current time %f is written' % current_time)
        print('-'*50)
    tnse_physics.write(time_stamp=current_time)

    # Solve
    while current_time < time_span:

        just_written = False
        early_exit = False

        # Update time step
        time_step += 1
        current_time += dt
        try:
            tnse_physics_solver.solve()
        except: # any error at all at the solve stage will result in an early exit
            early_exit = True

        # Check for quick exit on any rank, all rank exit
        if tnse_physics.mesh.comm.allreduce(early_exit, MPI.LOR):
            print('-'*50)
            print('Current time %f'%current_time)
            warning('Solve step exit eairly in an ERROR.')
            print('-'*50)
            break

        tnse_physics.update_previous_solution()

        # Write solution
        if time_step%save_every == 0:
            tnse_physics.write(time_stamp=current_time)
            if tnse_physics.mesh.comm.rank == 0:
                print('-'*50)
                print('Current time %f is written' % current_time)
                print('-'*50)
            tnse_physics.mesh.comm.Barrier()
            just_written = True

    # Write final time step
    if not just_written:
        if tnse_physics.mesh.comm.rank == 0:
            print('-'*50)
            print('Current time %f is written' % current_time)
            print('-'*50)
        tnse_physics.write(time_stamp=current_time)

def main(input_file):

    # Read input object
    input_object = InputObject(input_file)

    # Get physics
    physics = build_physics(input_object)

    # Build la solver
    la_solver = build_la_solver(input_object)

    # Set physics solver
    physics_solver = PhysicsSolver(physics, la_solver)

    # Solve and write result
    if input_object('flow physics type').lower() == 'stokes':
        _solve_stk(input_object, physics_solver, physics)
    elif input_object('flow physics type').lower() == 'steady navier stokes':
        _solve_snse(input_object, physics_solver, physics)
    else:
        _solve_tnse(input_object, physics_solver, physics)

if __name__ == '__main__':
    input_file = sys.argv[1]
    main(input_file)







