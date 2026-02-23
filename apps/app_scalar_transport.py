import flatiron_tk
import sys

from common import *
from flatiron_tk.io import InputObject
from flatiron_tk.mesh import Mesh
from flatiron_tk.physics import SteadyScalarTransport
from flatiron_tk.physics import TransientScalarTransport
from mpi4py import MPI

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

        bc_value = flatiron_tk.constant(physics.mesh, bc_value)

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
    if mesh is None: mesh = Mesh(mesh_file=mesh_file)
    else: mesh = mesh

    # Create physics object
    physics_type = input_object('transport physics type').lower()

    if physics_type == 'steady scalar transport'.lower():
        if mesh.comm.rank == 0:
            print('Building steady scalar transport physics')
        physics = SteadyScalarTransport(mesh)
    elif physics_type == 'transient scalar transport'.lower():
        if mesh.comm.rank == 0:
            print('Building transient scalar transport physics')
        dt = input_object('time step size')
        physics = TransientScalarTransport(mesh, dt, theta=0.5)
    else:
        emsg = custom_err_msg(physics_type, 'flow physics type', ['steady scalar transport', 'transient scalar transport'])
        raise ValueError(emsg)

    # Build function space
    physics.set_element('CG', 1)
    physics.set_tag('c')
    physics.build_function_space()

    # Set other parameters
    D = input_object('diffusivity')
    u = input_object('flow velocity')
    r = input_object('reaction')

    if isinstance(u, tuple):
        u = flatiron_tk.constant(mesh, u)

    if physics_type == 'steady scalar transport'.lower():
        physics.set_diffusivity(D)
        physics.set_advection_velocity(u)
        physics.set_reaction(r)
    elif physics_type == 'transient scalar transport'.lower():
        physics.set_diffusivity(D, D)
        physics.set_advection_velocity(u, u)
        physics.set_reaction(r, r)

    # Set weak form 
    physics.set_weak_form()

    # Set bcs 
    bc_dict = _parse_bc_input(input_object, physics)
    
    physics.set_bcs(bc_dict)

    # Set writer
    output_prefix = input_object('output prefix')
    output_type = input_object('output type')
    physics.set_writer(output_prefix, output_type)
    return physics


def _solve_steady(input_object, solver, physics):
    solver.solve()
    physics.write()

def _solve_transient(input_object, solver, physics):
    # Set time
    dt = input_object('time step size')
    time_span = input_object('time span')
    save_every = input_object('save every')
    current_time = 0
    time_step = 0

    ic = input_object('initial condition')
    # If there is an initial condition, read it in and interpolate to the function space
    if ic is not None:
        ic_file = ic[0]
        ic_name = ic[1]

        if len(ic) == 3:
            ic_time_id = ic[2]
        else:
            ic_time_id = 0

            f = flatiron_tk.bp_read_function(ic_file, time_id=ic_time_id, name=ic_name, element_family='CG', element_degree=1, element_shape='scalar')
            
            V = physics.get_function_space()
            c0 = flatiron_tk.interpolate_nonmatching(f, V)
            physics.set_initial_condition(c0)

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

        if physics.mesh.comm.allreduce(early_exit, MPI.LOR):
            print('-'*50)
            print('Current time %f'%current_time)
            flatiron_tk.custom_warning_message('Solve step exit eairly in an ERROR.')
            print('-'*50)
            break

        physics.update_previous_solution()

        # Write 
        if time_step % save_every == 0:
            physics.write(time_stamp=current_time)

            if physics.mesh.comm.rank == 0:
                print('-'*50)
                print(f'Current time {current_time:.4f} written to file')
                print('-'*50)
            just_written = True

    # Write the final time step 
    if not just_written:
        physics.write(time_stamp=current_time)
        if physics.mesh.comm.rank == 0:
            print('-'*50)
            print(f'Current time {current_time:.4f} written to file')
            print('-'*50)

def main(input_file):
    input_object = InputObject(input_file)

    # Create physics object
    physics = build_physics(input_object)
    solver = build_solver(physics, input_object)

    if isinstance(physics, TransientScalarTransport):
        _solve_transient(input_object, solver, physics)
    
    elif isinstance(physics, SteadyScalarTransport):
        _solve_steady(input_object, solver, physics)


if __name__ == '__main__':
    input_file = sys.argv[1]
    main(input_file)