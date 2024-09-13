import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

from flatiron_tk.io import InputObject
from flatiron_tk.info import info, warning, error
from flatiron_tk.physics import ScalarTransport
from flatiron_tk.physics import TransientScalarTransport
from flatiron_tk.io import h5_mod
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import PhysicsSolver
from flatiron_tk.functions.profiles import parabolic_2d, parabolic_3d, plug
import fenics as fe
from mpi4py import MPI
from common import build_la_solver, custom_err_msg, to_fe_constant

def _parse_bc_input(input_object, physics):

    """
    bc_input will be (bnd_id, bc_type, bc_value)
    bc_type are either ``fixed value`` or `gradient value`
    """

    avail_bc_type = ['fixed value', 'gradient value']

    bc_dict = {}
    for key in input_object.input_dict.keys():
        if not key.startswith('BC'):
            continue

        # bc_input will be (bnd_id, bc_type, options....)
        bc_input = input_object(key)
        bc_input = list(bc_input)
        bnd_id = int(bc_input[0])
        bc_type = bc_input[1]
        bc_value = input_object(bc_input[2])
        if isinstance(bc_value, tuple): bc_value = fe.Constant(bc_value)

        if bc_type == 'fixed value':
            bc_dict[bnd_id] = {'type': 'dirichlet', 'value': bc_value}
        elif bc_type in 'gradient value':
            bc_dict[bnd_id] = {'type': 'neumann', 'value': bc_value}
        else:
            emsg = custom_err_msg(bc_type, 'boundary condition type', avail_bc_type)
            raise ValueError(emsg)

    return bc_dict

def build_physics(input_object, mesh=None):

    # Load mesh
    mesh_file = input_object('mesh file')
    if mesh is None:
        mesh = Mesh(mesh_file=mesh_file)
    else:
        mesh = mesh

    # Build physics
    physics_type = input_object('transport physics type').lower()
    if physics_type == 'steady adr':
        physics = ScalarTransport(mesh)
    elif physics_type == 'transient adr':
        dt = input_object('time step size')
        physics = TransientScalarTransport(mesh, dt, theta=0.5)
    else:
        emsg = custom_err_msg(physics_type, 'flow physics type', ['steady adr', 'transient adr'])
        raise ValueError(emsg)

    # Build function space
    physics.set_element('CG', 1)
    physics.build_function_space()

    # Set other parameters
    D = input_object('diffusivity')
    u = input_object('flow velocity')
    if isinstance(u, tuple): u = fe.Constant(u)
    r = input_object('reaction')
    if physics_type == 'steady adr':
        physics.set_diffusivity(D)
        physics.set_advection_velocity(u)
        physics.set_reaction(r)
    else:
        physics.set_diffusivity(D, D)
        physics.set_advection_velocity(u, u)
        physics.set_reaction(r, r)

    # set weak form
    physics.set_weak_form()
    if input_object('add supg'):
        physics.add_stab()

    # Set bcs
    bc_dict = _parse_bc_input(input_object, physics)
    physics.set_bcs(bc_dict)

    # Set writer
    output_prefix = input_object('output prefix')
    output_type = input_object('output type')
    physics.set_writer(output_prefix, output_type)
    return physics

def _solve_sadr(input_object, adr_physics_solver, adr_physics):
    adr_physics_solver.solve()
    adr_physics.write()

def _solve_tadr(input_object, tadr_physics_solver, tadr_physics):

    # Set time
    dt = input_object('time step size')
    time_span = input_object('time span')
    save_every = input_object('save every')
    current_time = 0
    time_step = 0

    # Set initial condition
    ic = input_object('initial condition')
    if ic is not None:
        ic_file = ic[0]
        ic_group = ic[1]
        if len(ic) == 3: 
            ic_time_id = ic[2]
        else:
            ic_time_id = None
        ic_func = tadr_physics.read_function_from_h5(ic_file, ic_group, ic_time_id)
        tadr_physics.set_initial_condition(ic_func)
    else:
        '''Do nothing, assumes zero bc'''


    # Save initial condition
    if tadr_physics.mesh.comm.rank == 0:
        print('-'*50)
        print('Current time %f is written' % current_time)
        print('-'*50)
    tadr_physics.write(time_stamp=current_time)

    # Solve
    while current_time < time_span:

        just_written = False
        early_exit = False

        # Update time step
        time_step += 1
        current_time += dt
        try:
            tadr_physics_solver.solve()
        except: # any error at all at the solve stage will result in an early exit
            early_exit = True

        # Check for quick exit on any rank, all rank exit
        if tadr_physics.mesh.comm.allreduce(early_exit, MPI.LOR):
            print('-'*50)
            print('Current time %f'%current_time)
            warning('Solve step exit eairly in an ERROR.')
            print('-'*50)
            break

        # Update previous solution
        tadr_physics.update_previous_solution()

        # Write solution
        if time_step%save_every == 0:
            tadr_physics.write(time_stamp=current_time)
            if tadr_physics.mesh.comm.rank == 0:
                print('-'*50)
                print('Current time %f is written' % current_time)
                print('-'*50)
            tadr_physics.mesh.comm.Barrier()
            just_written = True

    # Write final time step
    if not just_written:
        if tadr_physics.mesh.comm.rank == 0:
            print('-'*50)
            print('Current time %f is written' % current_time)
            print('-'*50)
        tadr_physics.write(time_stamp=current_time)

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
    if input_object('transport physics type').lower() == 'steady adr':
        _solve_sadr(input_object, physics_solver, physics)
    elif input_object('transport physics type').lower() == 'transient adr':
        _solve_tadr(input_object, physics_solver, physics)
    else:
        custom_err_msg(input_object('transport physics type'), 'transport physics', ['steady adr', 'transient adr'])
        raise ValueError(emsg)

if __name__ == '__main__':
    input_file = sys.argv[1]
    main(input_file)







