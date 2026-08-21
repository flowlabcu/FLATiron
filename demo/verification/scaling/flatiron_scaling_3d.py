"""
flatiron_tk-based Poisson strong/weak scaling probe -- 3D unit-cube /
tetrahedron version, replicating the actual problem solved by
FEniCS/performance-test's poisson::problem() (src/poisson_problem.cpp,
src/Poisson.py, src/mesh.cpp on the `main` branch):

  - Unit cube, tetrahedra, CG1.
  - a(u,v) = inner(grad(u), grad(v))*dx
  - L(v)   = f*v*dx + g*v*ds, with
        f = 10*exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)  (Gaussian, interpolated)
        g = sin(5*x)                                    (interpolated)
  - Dirichlet u=0 on x=0 and x=1 only (the other 4 faces carry the g*v*ds
    Neumann flux instead of a Dirichlet condition).
  - Quadrature degree: flatiron_tk's PhysicsProblem base class always fixes
    quadrature_degree=4 (its own library default, not something set here);
    the reference lets FFCx pick automatically. For a degree-1 Lagrange
    element this has negligible effect on the stiffness matrix (its
    integrand is piecewise constant) and only mildly affects the load
    vector's numerical accuracy, not assembly/solve cost, so it does not
    materially affect scaling timings. dolfinx_scaling_3d.py matches this
    same quadrature_degree=4 so the two implementations stay apples-to-
    apples with each other.
  - CG + hypre BoomerAMG with strong_threshold=0.7, agg_nl=4,
    agg_num_paths=2, ksp_rtol=1e-8 (.github/workflows/ccpp.yml).

Mirrors `flatiron_scaling.py` (the 2D version) stage-for-stage. Companion
script `dolfinx_scaling_3d.py` mirrors this using raw dolfinx/ufl/petsc4py.

Usage:
    mpirun -n <N> python3 flatiron_scaling_3d.py
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import dolfinx
import numpy as np
import sys
import time

from mpi4py import MPI
from petsc4py import PETSc

from flatiron_tk.mesh import CuboidMesh
from flatiron_tk.physics import Poisson
from flatiron_tk.solver import LinearProblem
from flatiron_tk.solver import LinearSolver

comm = MPI.COMM_WORLD
NUM_RANKS = comm.size

# Elements per side for a single rank in the weak-scaling case. NE_BASE=78
# -> (79)^3 = 493,039 DOFs at 1 rank, matching the reference's ~500k
# DOFs/rank weak-scaling granularity.
NE_BASE = 78


def _stage(name, ne):
    # Flushed, rank-0-only stage marker so a hang shows up in the .out log
    if comm.rank == 0:
        print(f'[ne={ne}, ranks={NUM_RANKS}] entering: {name}', flush=True)
    sys.stdout.flush()


def _interpolate_f(x):
    # Gaussian source, independent of z (matches reference's f coefficient).
    return 10 * np.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)


def _interpolate_g(x):
    # Neumann flux coefficient (matches reference's g coefficient).
    return np.sin(5 * x[0])


def _poisson_ksp_setup(ksp):
    # Reference benchmark's own 3D-tuned hypre BoomerAMG settings
    # (.github/workflows/ccpp.yml in FEniCS/performance-test).
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix() or ''
    opts[f'{prefix}ksp_type'] = 'cg'
    opts[f'{prefix}pc_type'] = 'hypre'
    opts[f'{prefix}pc_hypre_type'] = 'boomeramg'
    opts[f'{prefix}pc_hypre_boomeramg_strong_threshold'] = 0.5
    opts[f'{prefix}pc_hypre_boomeramg_coarsen_type'] = 'hmis'
    opts[f'{prefix}pc_hypre_boomeramg_agg_nl'] = 4
    opts[f'{prefix}pc_hypre_boomeramg_agg_num_paths'] = 2
    opts[f'{prefix}ksp_rtol'] = 1e-8
    opts[f'{prefix}ksp_atol'] = 1e-50
    opts[f'{prefix}ksp_divtol'] = 1e4
    opts[f'{prefix}ksp_max_it'] = 1000
    ksp.setFromOptions()

    assert ksp.getType() == 'cg' and ksp.getPC().getType() == 'hypre', \
        'KSP options were not applied as expected'


def _solve_once(ne):
    """
    Build and solve one Poisson problem on an ne x ne x ne unit-cube mesh,
    timing each stage separately.

    Returns
    -------
    dict with keys: ne, dofs_global, mesh_s, setup_s, assemble_s,
    pc_setup_s, solve_s, ksp_iters
    """
    _stage('mesh', ne)
    comm.barrier()
    t0 = time.time()
    mesh = CuboidMesh(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1 / ne,
                    ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
                    partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet))
    comm.barrier()
    mesh_s = time.time() - t0

    _stage('function space / bcs setup', ne)
    comm.barrier()
    t0 = time.time()
    poisson = Poisson(mesh)
    poisson.set_element('CG', 1)
    poisson.build_function_space()
    V = poisson.get_function_space()

    f = dolfinx.fem.Function(V)
    f.interpolate(_interpolate_f)
    poisson.set_source(f)
    poisson.set_weak_form()

    g = dolfinx.fem.Function(V)
    g.interpolate(_interpolate_g)

    # Dirichlet u=0 on x=0 and x=1 (markers 1, 4)
    zero_u = dolfinx.fem.Function(V)
    zero_u.x.array[:] = 0
    zero_u.x.scatter_forward()

    bc_dict = {1: {'type': 'dirichlet', 'value': zero_u},
               4: {'type': 'dirichlet', 'value': zero_u},
               2: {'type': 'neumann', 'value': g},
               3: {'type': 'neumann', 'value': g},
               5: {'type': 'neumann', 'value': g},
               6: {'type': 'neumann', 'value': g}}
    poisson.set_bcs(bc_dict)
    comm.barrier()
    setup_s = time.time() - t0

    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    problem = LinearProblem(poisson)
    solver = LinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=_poisson_ksp_setup)

    _stage('assemble', ne)
    comm.barrier()
    t0 = time.time()
    solver.assemble()
    comm.barrier()
    assemble_s = time.time() - t0

    _stage('pc_setup (BoomerAMG hierarchy)', ne)
    comm.barrier()
    t0 = time.time()
    solver.krylov_solver.setUp()  # builds the BoomerAMG hierarchy, zero Krylov iterations
    comm.barrier()
    pc_setup_s = time.time() - t0

    _stage('solve', ne)
    comm.barrier()
    t0 = time.time()
    solver.krylov_solver.solve(solver._b, poisson.solution.x.petsc_vec)
    comm.barrier()
    solve_s = time.time() - t0
    poisson.solution.x.scatter_forward()

    ksp_iters = solver.krylov_solver.getIterationNumber()
    _stage('done', ne)

    return {
        'ne': ne,
        'dofs_global': num_dofs_global,
        'mesh_s': mesh_s,
        'setup_s': setup_s,
        'assemble_s': assemble_s,
        'pc_setup_s': pc_setup_s,
        'solve_s': solve_s,
        'ksp_iters': ksp_iters,
    }

def _warm_up():
    # Untimed warm-up solve on a tiny mesh, so JIT/form-compilation cost
    _solve_once(ne=32)

def strong_scaling():
    ne = 78  # fixed mesh size regardless of rank count
    result = _solve_once(ne)
    result['num_ranks'] = NUM_RANKS
    result['dofs_per_rank'] = result['dofs_global'] / NUM_RANKS
    return result

def weak_scaling():
    # Mesh refined so cells-per-rank (and therefore DOFs-per-rank) stays
    # ~constant as rank count grows (cube root, since this is a 3D volume).
    ne = round(NE_BASE * NUM_RANKS ** (1 / 3))
    result = _solve_once(ne)
    result['num_ranks'] = NUM_RANKS
    result['dofs_per_rank'] = result['dofs_global'] / NUM_RANKS
    return result

def _write_row(csv_path, row):
    if comm.rank != 0:
        return
    columns = ['num_ranks', 'ne', 'dofs_global', 'dofs_per_rank',
               'mesh_s', 'setup_s', 'assemble_s', 'pc_setup_s', 'solve_s',
               'ksp_iters']
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if write_header:
            f.write(', '.join(columns) + '\n')
        f.write(', '.join(
            f'{row[c]:.6f}' if isinstance(row[c], float) else str(row[c])
            for c in columns
        ) + '\n')


if __name__ == '__main__':
    # Optional suffix (e.g. "_multinode") so a differently-configured sweep
    # (different node/task topology) doesn't overwrite the default CSVs.
    suffix = os.environ.get('SCALING_CSV_SUFFIX', '')

    _warm_up()

    # strong_row = strong_scaling()
    # _write_row(f'flatiron_strong_scaling_3d{suffix}.csv', strong_row)

    weak_row = weak_scaling()
    _write_row(f'flatiron_weak_scaling_3d{suffix}.csv', weak_row)
