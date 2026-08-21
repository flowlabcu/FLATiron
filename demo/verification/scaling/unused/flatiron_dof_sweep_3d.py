"""
flatiron_tk-based Poisson granularity sweep -- 3D unit-cube / tetrahedron,
same problem/KSP settings as flatiron_scaling_3d.py (strong_threshold=0.7,
agg_nl=4, agg_num_paths=2, ghost_mode=none, matching the FEniCS/
performance-test reference), but instead of holding DOFs/rank fixed and
varying rank count, this holds rank count fixed (whatever it's launched
with) and sweeps DOFs/rank across TARGET_DOFS_PER_RANK.

Run this at rank=1 to get the baseline, then at some rank=N, so
plot_dof_efficiency_3d.py can compute efficiency = t[1]/t[N] separately at
each granularity and show how efficiency depends on problem size per rank
(distinct from how it depends on rank count, which is what the *_scaling_3d
scripts measure).

Usage:
    mpirun -n <N> python3 flatiron_dof_sweep_3d.py
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import dolfinx
import numpy as np
import time
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from flatiron_tk.mesh import CuboidMesh
from flatiron_tk.physics import Poisson
from flatiron_tk.solver import LinearProblem
from flatiron_tk.solver import LinearSolver

comm = MPI.COMM_WORLD
NUM_RANKS = comm.size

TARGET_DOFS_PER_RANK = [50_000, 100_000, 200_000, 500_000, 1_000_000]


def _ne_for_target_dofs(target_total):
    # P1 tetrahedra on a structured (ne+1)^3 vertex grid -> dofs = (ne+1)^3.
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
    mesh = CuboidMesh(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1 / ne,
                       ghost_mode=dolfinx.mesh.GhostMode.none)
    comm.barrier()
    mesh_s = time.time() - t0

    comm.barrier()
    t0 = time.time()
    poisson = Poisson(mesh)
    poisson.set_element('CG', 1)
    poisson.build_function_space()

    x = ufl.SpatialCoordinate(mesh.msh)
    poisson.set_source(body_force(x))
    poisson.set_weak_form()

    zero_u = dolfinx.fem.Function(poisson.get_function_space())
    zero_u.x.array[:] = 0
    zero_u.x.scatter_forward()

    bc_dict = {i: {'type': 'dirichlet', 'value': zero_u} for i in range(1, 7)}
    poisson.set_bcs(bc_dict)
    comm.barrier()
    setup_s = time.time() - t0

    V = poisson.get_function_space()
    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    problem = LinearProblem(poisson)
    solver = LinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=_poisson_ksp_setup)

    comm.barrier()
    t0 = time.time()
    solver.assemble()
    comm.barrier()
    assemble_s = time.time() - t0

    comm.barrier()
    t0 = time.time()
    solver.krylov_solver.setUp()
    comm.barrier()
    pc_setup_s = time.time() - t0

    comm.barrier()
    t0 = time.time()
    solver.krylov_solver.solve(solver._b, poisson.solution.x.petsc_vec)
    comm.barrier()
    solve_s = time.time() - t0
    poisson.solution.x.scatter_forward()

    ksp_iters = solver.krylov_solver.getIterationNumber()

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
        _write_row('flatiron_dof_sweep_3d.csv', result)
