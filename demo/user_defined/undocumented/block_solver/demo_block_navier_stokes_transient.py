import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

import fenics as fe
from petsc4py import PETSc
from flatiron_tk.mesh import Mesh
from flatiron_tk.physics import ScalarTransport, MultiPhysicsProblem
from flatiron_tk.physics import StokesFlow, SteadyIncompressibleNavierStokes, IncompressibleNavierStokes
from flatiron_tk.solver import PhysicsSolver, NonLinearProblem
from collections.abc import Iterable
from mpi4py import MPI
from flatiron_tk.solver import ConvergenceMonitor
from flatiron_tk.solver import BlockNonLinearSolver, FieldSplitTree

def get_avg_cell_diameter(mesh):

    hs = [c.h() for c in fe.cells(mesh.mesh)]
    h_sum = np.sum(hs)
    num_cell = len(hs)
    h_sum = MPI.COMM_WORLD.allgather(h_sum)
    num_cell = MPI.COMM_WORLD.allgather(num_cell)
    h_avg = np.sum(h_sum)/np.sum(num_cell)
    return h_avg

def build_bfs_problem(Re, CFL, mesh):

    rho = 1
    U = 10
    Re = Re
    inlet_face_size = 2
    mu = rho*U*inlet_face_size/Re
    avg_cell_diameter = get_avg_cell_diameter(mesh)
    dt = CFL*avg_cell_diameter/U
    print("dt =", dt)

    # Set physics problem
    nse = IncompressibleNavierStokes(mesh)
    nse.set_element('CG', 1, 'CG', 1)
    nse.build_function_space()
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)
    nse.set_time_step_size(dt)
    nse.set_mid_point_theta(0.5)
    nse.set_weak_form()
    nse.add_stab()

    # Set boundary conditions
    zero_v = fe.Constant( (0, 0) )
    zero = fe.Constant(0)
    inlet = fe.Expression( ("sin(2*pi*t)*U*x[1]*(2-x[1])", "0"), t=0, U=U, degree=2 )
    u_bcs = {
            1: {'type': 'dirichlet', 'value': inlet},
            2: {'type': 'dirichlet', 'value': zero_v},
            3: {'type': 'dirichlet', 'value': zero_v},
            4: {'type': 'dirichlet', 'value': zero_v},
            6: {'type': 'dirichlet', 'value': zero_v}
            }
    p_bcs = {5: {'type': 'dirichlet', 'value': zero}}
    bc_dict = {'u': u_bcs, 'p': p_bcs}
    nse.set_bcs(bc_dict)

    # Set initial guess to be non-zero so that
    # we don't have a divide by 0 error
    eps = fe.DOLFIN_EPS
    u_ini = fe.interpolate( fe.Constant( (eps, eps) ), nse.V.sub(0).collapse() )
    p_ini = fe.interpolate( fe.Constant(eps), nse.V.sub(1).collapse() )
    nse.set_initial_guess(u_ini, p_ini)
    for bc in nse.dirichlet_bcs:
        bc.apply(nse.solution.vector())
    return nse, inlet

def main():

    Re = 1000
    CFL = 1
    mesh = Mesh(mesh_file='/home/cteerara/Workspace-TMP/petsc_block_precond/mesh/h5/bfs_l3.h5')
    nse, inlet = build_bfs_problem(Re, CFL, mesh)
    problem = NonLinearProblem(nse)

    def set_ksp_u(ksp):
        ksp.setType(PETSc.KSP.Type.FGMRES)
        ksp.setMonitor(ConvergenceMonitor("|----KSP0", verbose=False))
        ksp.setTolerances(max_it=5)
        ksp.pc.setType(PETSc.PC.Type.JACOBI)
        ksp.setUp()

    def set_ksp_p(ksp):
        ksp.setType(PETSc.KSP.Type.FGMRES)
        ksp.setMonitor(ConvergenceMonitor("|--------KSP1", verbose=False))
        ksp.setTolerances(max_it=3)
        ksp.pc.setType(PETSc.PC.Type.HYPRE)
        ksp.pc.setHYPREType("boomeramg")
        ksp.setUp()

    def set_outer_ksp(ksp):
        ksp.setType(PETSc.KSP.Type.FGMRES)
        ksp.setGMRESRestart(50)
        ksp.setTolerances(rtol=1e-100, atol=1e-10)
        ksp.setMonitor(ConvergenceMonitor("Outer ksp"))

    split = {'fields': ('u', 'p'),
             'composite_type': 'schur',
             'schur_fact_type': 'full',
             'schur_pre_type': 'a11',
             'ksp0_set_function': set_ksp_u,
             'ksp1_set_function': set_ksp_p}

    tree = FieldSplitTree(nse, split)
    solver = BlockNonLinearSolver(tree, mesh.comm, 
                                  problem, fe.PETScKrylovSolver(),
                                  outer_ksp_set_function=set_outer_ksp)

    T = 2
    nt = int(T/nse.external_function('dt').values()[0]) + 1
    nse.set_writer('output', 'h5')
    t = 0
    for i in range(nt):

        t += nse.external_function('dt').values()[0]
        inlet.t = t

        solver.solve()
        nse.update_previous_solution()
        nse.write(time_stamp=t)

        if nse.mesh.comm.rank==0:
            print("-"*50)
            print("Time step solved: %d/%d"%(i+1, nt))
            print("-"*50)
            print("\n")

if __name__ == '__main__':
    main()
