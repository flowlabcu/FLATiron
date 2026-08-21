"""
Bare-dolfinx Poisson strong/weak scaling probe -- 3D unit-cube / tetrahedron
version, replicating the actual problem solved by FEniCS/performance-test's
poisson::problem() (src/poisson_problem.cpp, src/Poisson.py, src/mesh.cpp on
the `main` branch):

  - Unit cube, tetrahedra, CG1.
  - a(u,v) = inner(grad(u), grad(v))*dx
  - L(v)   = f*v*dx + g*v*ds, with
        f = 10*exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)  (Gaussian, interpolated)
        g = sin(5*x)                                    (interpolated)
  - Dirichlet u=0 on x=0 and x=1 only (the other 4 faces carry the g*v*ds
    Neumann flux instead of a Dirichlet condition).
  - quadrature_degree=4 kept explicit here (not the reference's automatic
    FFCx default) so this stays apples-to-apples with flatiron_scaling_3d.py,
    whose PhysicsProblem base class always fixes quadrature_degree=4. See
    that file's docstring for why this doesn't materially affect timings.
  - CG + hypre BoomerAMG with strong_threshold=0.7, agg_nl=4,
    agg_num_paths=2, ksp_rtol=1e-8 (.github/workflows/ccpp.yml).

Mirrors `dolfinx_scaling.py` (the 2D version) stage-for-stage; NO
flatiron_tk imports.

Usage:
    mpirun -n <N> python3 dolfinx_scaling_3d.py
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import sys
import time
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import os
# Force PETSc's matrix layout and mesh structures to use a strict geometric Cartesian topology
os.environ["PETSc_OPTIONS"] = "-mat_dm_amatrix_distribute true -pc_hypre_boomeramg_coarsen_type pmis -pc_hypre_boomeramg_strong_threshold 0.5"


comm = MPI.COMM_WORLD
NUM_RANKS = comm.size

# Elements per side for a single rank in the weak-scaling case. NE_BASE=78
# -> (79)^3 = 493,039 DOFs at 1 rank, matching the reference's ~500k
# DOFs/rank weak-scaling granularity.
NE_BASE = 78


def _stage(name, ne):
    if comm.rank == 0:
        print(f'[ne={ne}, ranks={NUM_RANKS}] entering: {name}', flush=True)
    sys.stdout.flush()


def _interpolate_f(x):
    # Gaussian source, independent of z
    return 10 * np.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)


def _interpolate_g(x):
    # Neumann flux coefficient
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
    timing each stage separately. Same return-dict shape as the 2D script.
    """
    _stage('mesh', ne)
    comm.barrier()
    t0 = time.time()
    msh = dolfinx.mesh.create_box(
        comm, [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [ne, ne, ne], dolfinx.mesh.CellType.tetrahedron,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet))

    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)

    # 6 cube faces tagged independently, mirroring CuboidMesh.mark_boundary.
    face_markers = {
        1: lambda x: np.isclose(x[0], 0.0),
        2: lambda x: np.isclose(x[1], 0.0),
        3: lambda x: np.isclose(x[2], 0.0),
        4: lambda x: np.isclose(x[0], 1.0),
        5: lambda x: np.isclose(x[1], 1.0),
        6: lambda x: np.isclose(x[2], 1.0),
    }
    entity_ids, marking_ids = [], []
    for idx, marker in face_markers.items():
        found = dolfinx.mesh.locate_entities(msh, fdim, marker)
        entity_ids.extend(found)
        marking_ids.extend([idx] * len(found))
    entity_ids = np.array(entity_ids)
    marking_ids = np.array(marking_ids)
    order = np.argsort(entity_ids)
    boundary_tags = dolfinx.mesh.meshtags(msh, fdim, entity_ids[order], marking_ids[order])

    comm.barrier()
    mesh_s = time.time() - t0

    _stage('function space / bcs setup', ne)
    comm.barrier()
    t0 = time.time()
    V = dolfinx.fem.functionspace(msh, ('Lagrange', 1))
    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)

    f = dolfinx.fem.Function(V)
    f.interpolate(_interpolate_f)
    g = dolfinx.fem.Function(V)
    g.interpolate(_interpolate_g)

    dx = ufl.Measure('dx', metadata={'quadrature_degree': 4})
    # Full boundary, no subdomain restriction. 
    ds = ufl.Measure('ds', domain=msh, metadata={'quadrature_degree': 4})
    a = ufl.inner(ufl.grad(w), ufl.grad(u)) * dx
    L = f * w * dx + g * w * ds

    # Dirichlet u=0 on x=0 and x=1 (markers 1, 4) only; the other 4 faces
    # carry the g*v*ds Neumann flux instead.
    zero = dolfinx.fem.Function(V)
    zero.x.array[:] = 0.0
    bcs = []
    for idx in (1, 4):
        dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_tags.find(idx))
        bcs.append(dolfinx.fem.dirichletbc(zero, dofs))
    comm.barrier()
    setup_s = time.time() - t0

    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    a_form = dolfinx.fem.form(a)
    L_form = dolfinx.fem.form(L)

    ksp = PETSc.KSP().create(comm)
    _poisson_ksp_setup(ksp)

    _stage('assemble', ne)
    comm.barrier()
    t0 = time.time()
    A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = dolfinx.fem.petsc.create_vector(L_form)
    with b.localForm() as bl:
        bl.set(0.0)
    dolfinx.fem.petsc.assemble_vector(b, L_form)
    dolfinx.fem.petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    comm.barrier()
    assemble_s = time.time() - t0

    ksp.setOperators(A)
    xsol = dolfinx.fem.Function(V)

    _stage('pc_setup (BoomerAMG hierarchy)', ne)
    comm.barrier()
    t0 = time.time()
    ksp.setUp()  # builds the BoomerAMG hierarchy, zero Krylov iterations
    comm.barrier()
    pc_setup_s = time.time() - t0

    _stage('solve', ne)
    comm.barrier()
    t0 = time.time()
    ksp.solve(b, xsol.x.petsc_vec)
    comm.barrier()
    solve_s = time.time() - t0
    xsol.x.scatter_forward()

    ksp_iters = ksp.getIterationNumber()
    ksp.destroy()
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

def strong_scaling():
    ne = 78
    result = _solve_once(ne)
    result['num_ranks'] = NUM_RANKS
    result['dofs_per_rank'] = result['dofs_global'] / NUM_RANKS
    return result


def weak_scaling():
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
    suffix = os.environ.get('SCALING_CSV_SUFFIX', '')

    # strong_row = strong_scaling()
    # _write_row(f'dolfinx_strong_scaling_3d{suffix}.csv', strong_row)

    weak_row = weak_scaling()
    _write_row(f'dolfinx_weak_scaling_3d{suffix}.csv', weak_row)
