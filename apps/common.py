import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

import fenics as fe
from flatiron_tk.io import h5_mod


def custom_err_msg(usr_input, input_type, avail_types):
    msg = '%s is an invalid %s\n' % (usr_input, input_type)
    msg += 'Available %s are:\n' % (input_type)
    for i, type_name in enumerate(avail_types):
        msg += '%d. %s\n' % ( (i+1), type_name )
    msg += ('*'*50)
    return msg

def build_la_solver(input_object):
    solver_type = input_object('solver type')
    if solver_type == 'direct':
        return fe.LUSolver()
    else:
        return _build_ksp(input_object)

def _build_ksp(input_object):

    # Build linear solver
    ksp_solver = fe.PETScKrylovSolver()
    solver_type = input_object('solver type')
    pc_type = input_object('pc type')
    if pc_type is None:
        print("pc type for krylov solver is not provided, flatiron_tk defaults to the jacobi solver")
        pc_type = 'jacobi'
    fe.PETScOptions.set("ksp_type", solver_type)
    fe.PETScOptions.set("pc_type", pc_type)
    fe.PETScOptions.set("ksp_monitor")

    # Set tolerances
    rel_tol = input_object('ksp relative tolerance')
    abs_tol = input_object('ksp absolute tolerance')
    max_itr = input_object('ksp maximum iterations')
    ksp_solver.parameters["relative_tolerance"] = rel_tol
    ksp_solver.parameters["absolute_tolerance"] = abs_tol
    ksp_solver.parameters["maximum_iterations"] = max_itr
    ksp_solver.parameters["monitor_convergence"] = input_object("ksp monitor convergence")

    # Look for petsc inputs and set these options if it's there
    for key in input_object.input_dict.keys():
        if not key.startswith('petsc options'):
            continue
        option = input_object(key)
        if isinstance(option, tuple):
            fe.PETScOptions.set(option[0], option[1])
        else:
            fe.PETScOptions.set(option)

    # Set uptions
    ksp_solver.set_from_options()
    return ksp_solver

def to_fe_constant(a):
    '''
    Convert a to fe.Constant(a) if a is a number or numpy array
    '''
    if isinstance(a, float) or isinstance(a, int):
        return fe.Constant(a)
    return a






