"""
Bare-dolfinx Poisson granularity sweep -- 3D unit-cube / tetrahedron, NO
flatiron_tk imports. Mirrors `flatiron_dof_sweep_3d.py` stage-for-stage; see
that file's docstring for the methodology (fixed rank count, sweep
DOFs/rank).

Usage:
    mpirun -n <N> python3 dolfinx_dof_sweep_3d.py
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import time
import ufl

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
NUM_RANKS = comm.size

TARGET_DOFS_PER_RANK = [50_000, 100_000, 200_000, 500_000, 1_000_000]


def _ne_for_target_dofs(target_total):
    ne = 1
    while (ne + 1) ** 3 < target_total:
        ne += 1
    if ne > 1 and abs(ne ** 3 - target_total) < abs((ne + 1) ** 3 - target_total):
        ne -= 1
    return ne


def body_force(x):
    return (3 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
            * ufl.sin(np.pi * x[2]))


def _poisson_ksp_setup(ksp):
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix() or ''
    opts[f'{prefix}ksp_type'] = 'cg'
    opts[f'{prefix}pc_type'] = 'hypre'
    opts[f'{prefix}pc_hypre_type'] = 'boomeramg'
    opts[f'{prefix}pc_hypre_boomeramg_strong_threshold'] = 0.7
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
    comm.barrier()
    t0 = time.time()
    msh = dolfinx.mesh.create_box(
        comm, [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [ne, ne, ne], dolfinx.mesh.CellType.tetrahedron,
        ghost_mode=dolfinx.mesh.GhostMode.none,
    )

    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)

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

    comm.barrier()
    t0 = time.time()
    V = dolfinx.fem.functionspace(msh, ('Lagrange', 1))
    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = body_force(x)

    dx = ufl.Measure('dx', metadata={'quadrature_degree': 4})
    a = ufl.inner(ufl.grad(w), ufl.grad(u)) * dx
    L = w * f * dx

    zero = dolfinx.fem.Function(V)
    zero.x.array[:] = 0.0
    bcs = []
    for idx in face_markers:
        dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_tags.find(idx))
        bcs.append(dolfinx.fem.dirichletbc(zero, dofs))
    comm.barrier()
    setup_s = time.time() - t0

    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    a_form = dolfinx.fem.form(a)
    L_form = dolfinx.fem.form(L)

    ksp = PETSc.KSP().create(comm)
    _poisson_ksp_setup(ksp)

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

    comm.barrier()
    t0 = time.time()
    ksp.setUp()
    comm.barrier()
    pc_setup_s = time.time() - t0

    comm.barrier()
    t0 = time.time()
    ksp.solve(b, xsol.x.petsc_vec)
    comm.barrier()
    solve_s = time.time() - t0
    xsol.x.scatter_forward()

    ksp_iters = ksp.getIterationNumber()
    ksp.destroy()

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
    _solve_once(ne=4)


def _write_row(csv_path, row):
    if comm.rank != 0:
        return
    columns = ['num_ranks', 'target_dofs_per_rank', 'ne', 'dofs_global',
               'dofs_per_rank', 'mesh_s', 'setup_s', 'assemble_s',
               'pc_setup_s', 'solve_s', 'ksp_iters']
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if write_header:
            f.write(', '.join(columns) + '\n')
        f.write(', '.join(
            f'{row[c]:.6f}' if isinstance(row[c], float) else str(row[c])
            for c in columns
        ) + '\n')


if __name__ == '__main__':
    _warm_up()

    for target in TARGET_DOFS_PER_RANK:
        ne = _ne_for_target_dofs(target * NUM_RANKS)
        result = _solve_once(ne)
        result['num_ranks'] = NUM_RANKS
        result['target_dofs_per_rank'] = target
        result['dofs_per_rank'] = result['dofs_global'] / NUM_RANKS
        _write_row('dolfinx_dof_sweep_3d.csv', result)
