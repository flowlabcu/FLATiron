from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver
from flatiron_tk.solver import ConvergenceMonitor
from petsc4py import PETSc

def custom_err_msg(usr_input, input_type, avail_types):
    msg = '%s is an invalid %s\n' % (usr_input, input_type)
    msg += 'Available %s are:\n' % (input_type)
    for i, type_name in enumerate(avail_types):
        msg += '%d. %s\n' % ( (i+1), type_name )
    msg += ('*'*50)
    return msg


def build_solver(physics, input_object):
    problem = NonLinearProblem(physics)
    ksp = build_ksp_from_input(input_object)
    solver = NonLinearSolver(physics.mesh.comm, problem, outer_ksp_set_function=ksp)
    return solver

def build_ksp_from_input(input_object):
    """
    Return a function that configures a PETSc KSP object
    based on parameters from the input_object.

    Usage:
        solver = NonLinearSolver(comm, problem,
                                 outer_ksp_set_function=build_ksp_from_input(input_object))
    """
    def _ksp_setup(ksp: PETSc.KSP):
        # Pull solver type
        solver_type = input_object('ksp type') or 'fgmres'
        pc_type     = input_object('pc type') or 'ilu'

        rel_tol = input_object('ksp relative tolerance') or 1e-8
        abs_tol = input_object('ksp absolute tolerance') or 1e-10
        max_it  = input_object('ksp maximum iterations') or 1000

        # Use PETSc.Options so solver can still be overridden by -ksp_* flags
        prefix = ksp.getOptionsPrefix()
        opts = PETSc.Options()

        opts[f"{prefix}ksp_type"]  = solver_type
        opts[f"{prefix}pc_type"]   = pc_type
        opts[f"{prefix}ksp_rtol"]  = rel_tol
        opts[f"{prefix}ksp_atol"]  = abs_tol
        opts[f"{prefix}ksp_max_it"] = max_it

        # Apply options to this KSP
        ksp.setFromOptions()

        # Add a convergence monitor (optional)
        ksp.setMonitor(ConvergenceMonitor('ksp'))

    return _ksp_setup
