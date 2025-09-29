
def import_dolfinx():
    dolfinx = None
    try:
        import dolfinx
    except ImportError:
        print('dolfinx not found. Please install dolfinx to use FLATiron_tk.')
    return dolfinx

def import_basix():
    basix = None
    try:
        import basix
    except ImportError:
        print('basix not found. Please install basix to use FLATiron_tk.')
    return basix

def import_adios4dolfinx():
    adios4dolfinx = None
    try:
        import adios4dolfinx
    except ImportError:
        print('adios4dolfinx not found. Please install adios4dolfinx to use FLATiron_tk.')
    return adios4dolfinx

def import_ufl():
    ufl = None
    try:
        import ufl
    except ImportError:
        print('ufl not found. Please install ufl to use FLATiron_tk.')
    return ufl

def import_PETSc():
    PETSc = None
    try:
        from petsc4py import PETSc
    except ImportError:
        print('petsc4py not found. Please install petsc4py to use FLATiron_tk.')
    return PETSc

def import_dolfinx_fem_petsc():
    petsc = None
    try:
        from dolfinx.fem import petsc
    except ImportError:
        print('dolfinx.fem.petsc not found. Please install dolfinx to use FLATiron_tk.')
    return petsc

def import_dolfinx_nls_petsc():
    petsc = None
    try:
        from dolfinx.nls import petsc
    except ImportError:
        print('dolfinx.nls.petsc not found. Please install dolfinx to use FLATiron_tk.')
    return petsc

def import_mpi4py():
    MPI = None
    try:
        from mpi4py import MPI
    except ImportError:
        print('mpi4py not found. Please install mpi4py to use FLATiron_tk.')
    return MPI