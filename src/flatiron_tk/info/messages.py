def import_fenics():
    '''
    Issue a warning instead of a straight up error
    for when you are importing fenics.
    '''
    fe = None
    try:
        import fenics as fe
    except ImportError:
        print("WARNING: unable to import FEniCS. Please make sure FEniCS is installed")
    return fe

def import_PETSc():
    PETSc = None
    try:
        from petsc4py import PETSc
    except ImportError:
        print("Warning: unable to import petsc4py. Please make sure petsc4py is installed")
    return PETSc

def info(msg, all_rank=False):
    fe = import_fenics()
    if all_rank:
        print(msg)
    else:
        if fe.MPI.comm_world.rank == 0:
            print(msg)

def warning(msg, all_rank=False):
    wrn_msg = 'WARNING: ' + msg
    info(wrn_msg, all_rank)

def error(err_type, err_msg):
    raise err_type(err_msg)
