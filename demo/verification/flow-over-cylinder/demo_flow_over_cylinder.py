import dolfinx
import ufl
import flatiron_tk
import numpy as np

from flatiron_tk.mesh import Mesh, Boundary
from flatiron_tk.physics import TransientNavierStokes
from flatiron_tk.solver  import ConvergenceMonitor
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import BlockNonLinearSolver, BlockSplitTree
from mpi4py import MPI
from petsc4py import PETSc

import csv
import matplotlib.pyplot as plt

def make_solver(nse):
    # Set the problem 
    problem = NonLinearProblem(nse)

    # Set up Block Solver 
    def set_ksp_u(ksp: PETSc.KSP):
        ksp.setType(PETSc.KSP.Type.FGMRES)
        ksp.pc.setType(PETSc.PC.Type.HYPRE)
        ksp.pc.setHYPREType("boomeramg")
        ksp.setTolerances(rtol=1e-5, atol=1e-8, max_it=50)
        ksp.monitorCancel()
        ksp.setMonitor(ConvergenceMonitor("|----KSP.U", verbose=True))

    def set_ksp_p(ksp: PETSc.KSP):
        ksp.setType(PETSc.KSP.Type.FGMRES)
        ksp.pc.setType(PETSc.PC.Type.HYPRE)
        ksp.pc.setHYPREType("boomeramg")
        ksp.monitorCancel()
        ksp.setMonitor(ConvergenceMonitor("|--------KSP.P", verbose=True))
        ksp.setTolerances(rtol=1e-3, atol=1e-8, max_it=50)

    # -----------------------------
    # Outer KSP
    # -----------------------------
    def set_outer_ksp(ksp: PETSc.KSP):
        ksp.setType(PETSc.KSP.Type.FGMRES)
        ksp.setGMRESRestart(50)
        ksp.setTolerances(rtol=1e-7, atol=1e-8)
        ksp.monitorCancel()
        ksp.setMonitor(ConvergenceMonitor("Outer ksp", verbose=True))


    split = {
        'fields': ('u', 'p'),
        'composite_type': 'schur',
        'schur_fact_type': 'upper',
        'schur_pre_type': 'selfp',
        'ksp0_set_function': set_ksp_u,
        'ksp1_set_function': set_ksp_p
    }

    # Build the block tree
    tree = BlockSplitTree(nse, splits=split)

    # Build the solver
    solver = BlockNonLinearSolver(
        tree,
        MPI.COMM_WORLD,
        problem,
        outer_ksp_set_function=set_outer_ksp,
        rtol=1e-6,
        atol=1e-8,
    )

    return solver

def run_flow_over_cylinder(reynolds_number=20):
    # Define the mesh
    mesh_file = 'mesh/foc.msh'
    mesh = Mesh(mesh_file=mesh_file)
    Inlet = Boundary(mesh, 1)
    Lower_wall = Boundary(mesh, 2)
    Outlet = Boundary(mesh, 3)
    Upper_wall = Boundary(mesh, 4)
    Cylinder = Boundary(mesh, 5)

    # Create transient Navier-Stokes object
    nse = TransientNavierStokes(mesh)
    nse.set_element('CG', 2, 'CG', 1)
    nse.build_function_space()

    # Physical parameters
    dt = flatiron_tk.constant(mesh, 0.01)
    mu = 0.001
    rho = 1

    D = Cylinder.area / np.pi  # cylinder diameter from circumference
    u_bar = reynolds_number * mu / (rho * D)
    flow_rate = u_bar * Inlet.radius * 2

    nse.set_time_step_size(dt)
    nse.set_midpoint_theta(0.5)
    nse.set_density(rho)
    nse.set_dynamic_viscosity(mu)
    nse.set_weak_form(stab=True)

    # Get function spaces for boundary conditions functions
    V_u = nse.get_function_space('u').collapse()[0]
    V_p = nse.get_function_space('p').collapse()[0]

    profile = flatiron_tk.ParabolicInletProfile(flow_rate=flow_rate, radius=Inlet.radius, center=Inlet.centroid, normal=-Inlet.normal)
    inlet_v = dolfinx.fem.Function(V_u)
    inlet_v.interpolate(profile)

    zero_p = dolfinx.fem.Function(V_p); zero_p.x.array[:] = 0.0; zero_p.x.scatter_forward()
    zero_v = dolfinx.fem.Function(V_u); zero_v.x.array[:] = 0.0; zero_v.x.scatter_forward()

    u_bcs = {Inlet.id: {'type': 'dirichlet', 'value': inlet_v},
            Lower_wall.id: {'type': 'dirichlet', 'value': zero_v},
            Cylinder.id: {'type': 'dirichlet', 'value': zero_v},
            Upper_wall.id: {'type': 'dirichlet', 'value': zero_v}}
    p_bcs = {Outlet.id: {'type': 'dirichlet', 'value': zero_p}}
    bc_dict = {'u': u_bcs, 'p': p_bcs}
    nse.set_bcs(bc_dict)

    # Set the output writer
    nse.set_writer('output', 'pvd')

    solver = make_solver(nse)

    #
    def _compute_drag_lift(u, p, id):
        n = mesh.get_facet_normal()
        stress_tensor = -p * ufl.Identity(len(u)) + 2*mu*ufl.sym(ufl.grad(u))

        traction = ufl.dot(stress_tensor, n)

        form_drag = dolfinx.fem.form(traction[0]*nse.ds(id))
        form_lift = dolfinx.fem.form(traction[1]*nse.ds(id))
        drag = dolfinx.fem.assemble_scalar(form_drag)
        lift = dolfinx.fem.assemble_scalar(form_lift)
        drag = mesh.comm.allreduce(drag, op=MPI.SUM)
        lift = mesh.comm.allreduce(lift, op=MPI.SUM)
        return drag, lift

    mu_const = nse.external_function('dynamic_viscosity')
    mus = np.logspace(np.log10(mu_const.value)+1, np.log10(mu_const.value), 10)
    for m in mus:
        mu_const.value = m
        solver.solve()
        nse.update_previous_solution()

    t = 0
    # Initialize 
    while t < 5.0:
        if mesh.comm.rank == 0:
            print(f'Solving warm-up at time t = {t:.2f}')
        # Solve the problem
        solver.solve()
        nse.update_previous_solution()
        # Update time
        t += dt.value

    # Now solve and compute drag/lift coefficients over time
    t = 0 
    dt.value = 0.005
    cd_array = []
    dl_array = []
    time_array = []
    while t < 0.35:  
        if mesh.comm.rank == 0: print(f'Solving at time t = {t:.2f}')

        
        solver.solve()
        nse.update_previous_solution()
        nse.write(time_stamp=t)

        uf = nse.solution.split()[0].collapse()
        pf = nse.solution.split()[1].collapse()

        drag, lift = _compute_drag_lift(uf, pf, Cylinder.id)
        cd = 2*drag/(rho*u_bar**2*D)
        cl = 2*lift/(rho*u_bar**2*D)
        
        time_array.append(t)
        cd_array.append(cd)
        dl_array.append(cl)

        t += dt.value

    cd_array = np.abs(np.array(cd_array))
    dl_array = np.array(dl_array)

    if mesh.comm.rank == 0:
        with open('drag_lift.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'Cd', 'Cl'])
            for t_val, cd_val, cl_val in zip(time_array, cd_array, dl_array):
                writer.writerow([f'{t_val:.6f}', f'{cd_val:.6e}', f'{cl_val:.6e}'])



if __name__ == '__main__':
    re = 100
    run_flow_over_cylinder(reynolds_number=re)