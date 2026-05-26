"""
Kovasznay Flow Verification
============================
Exact 2D steady NSE solution at Re=40 (Kovasznay, 1948).

    u(x,y) = 1 - exp(lambda*x) * cos(2*pi*y)
    v(x,y) = lambda/(2*pi) * exp(lambda*x) * sin(2*pi*y)
    p(x,y) = -exp(2*lambda*x) / 2

  lambda = Re/2 - sqrt(Re^2/4 + 4*pi^2)

Domain:  [-0.5, 1.5] x [-0.5, 1.5]
BCs:     Dirichlet velocity on all 4 walls (exact solution)
         Dirichlet pressure on left wall  (exact solution, constant there)

Expected convergence rates with P1/P1 + PSPG:
    velocity:  O(h^2)
    pressure:  O(h^1)

Reference:
    Kovasznay, L.S.G. (1948). Laminar flow behind a two-dimensional grid.
    Proc. Cambridge Phil. Soc., 44, 58-62.
"""

import numpy as np
import dolfinx
import pyvista
import ufl
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import pandas as pd

from flatiron_tk.mesh import RectMesh
from flatiron_tk.physics import SteadyNavierStokes
from flatiron_tk.solver import NonLinearProblem, NonLinearSolver
from flatiron_tk.info import custom_warning_message
import csv

def plot_solution(u_sol, h, lam):
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(u_sol.function_space.mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    block_shape = u_sol.function_space.dofmap.index_map_bs
    u_values = u_sol.x.array.reshape(-1, block_shape)

    if block_shape == 2:
        # 2D flow, pad with zeros in the Z direction
        u_3d = np.pad(u_values, ((0, 0), (0, 1)), mode='constant')
    else:
        u_3d = u_values

    grid.point_data['Velocity_3D'] = u_3d
    grid.point_data['Velocity Magnitude'] = np.linalg.norm(u_3d, axis=1)
    
    # Good practice to explicitly set active vectors for streamline generation
    grid.set_active_vectors('Velocity_3D')

    # Set up the background plotter
    plotter = pyvista.Plotter(off_screen=True, window_size=[1000, 800])
    plotter.add_text(f'Numerical Solution (lambda={lam:.1f}), h={h:.4f}', font_size=14)

    # Add the mesh colormap
    plotter.add_mesh(grid, scalars='Velocity Magnitude', cmap='viridis', show_scalar_bar=True)

    # Dynamic streamline seeding based on the bounds
    bounds = grid.bounds 
    seed_x = bounds[0] + 0.1 * (bounds[1] - bounds[0])
    
    # Use point a and point b to create a vertical 1D line exactly on the Z=0 plane
    streamlines = grid.streamlines(
        vectors='Velocity_3D',
        pointa=(seed_x, bounds[2] + 1e-4, 0.0),
        pointb=(seed_x, bounds[3] - 1e-4, 0.0),
        n_points=50,
        max_steps=5000 
    )
    
    # Safety check to ensure streamlines were generated before plotting
    if streamlines.n_points > 0:
        plotter.add_mesh(streamlines, color='white', line_width=2.0)
    else:
        custom_warning_message(f'Streamlines failed to generate any points for h={h}')

    plotter.view_xy()
    
    # Save the plot directly to the directory
    lam_str = 'lam1' if lam > 0 else 'lam2'
    plot_filename = f'images/plot_kovasznay_{lam_str}_h_{h:.4f}.png'
    plotter.screenshot(plot_filename)
    plotter.close() # Close to free up memory before the next loop iteration
    
def _style_ax(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12, width=1.5)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.figure.canvas.draw()
    ax.grid(True, alpha=0.2)

def plot_error(file):

    df = pd.read_csv(file)

    # Lambda 1
    df1 = df[df['lambda'] > 0].sort_values('h')
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.loglog(df1['h'], df1['L2_error_u'], 'o-', label='Velocity (u) Error', color='tab:blue', linewidth=2)
    ax1.loglog(df1['h'], df1['L2_error_p'], 's-', label='Pressure (p) Error', color='tab:red', linewidth=2)
    ref_u1 = (df1['h'] / df1['h'].iloc[-1])**2 * df1['L2_error_u'].iloc[-1]
    ax1.loglog(df1['h'], ref_u1, 'k--', label='O(h²)', alpha=0.7)
    ref_p1 = (df1['h'] / df1['h'].iloc[-1])**1 * df1['L2_error_p'].iloc[-1]
    ax1.loglog(df1['h'], ref_p1, 'k:', label='O(h¹)', alpha=0.7)
    ax1.set_xlabel('Mesh size ($h$)')
    ax1.set_ylabel('L2 Error')
    ax1.legend(fontsize=11)
    _style_ax(ax1)
    fig1.tight_layout()
    fig1.savefig('kovasznay_convergence_lam1.png', dpi=300)
    plt.close(fig1)

    # Lambda 2
    df2 = df[df['lambda'] < 0].sort_values('h')
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.loglog(df2['h'], df2['L2_error_u'], 'o-', label='Velocity (u) Error', color='tab:blue', linewidth=2)
    ax2.loglog(df2['h'], df2['L2_error_p'], 's-', label='Pressure (p) Error', color='tab:red', linewidth=2)
    ref_u2 = (df2['h'] / df2['h'].iloc[-1])**2 * df2['L2_error_u'].iloc[-1]
    ax2.loglog(df2['h'], ref_u2, 'k--', label='O(h²)', alpha=0.7)
    ref_p2 = (df2['h'] / df2['h'].iloc[-1])**1 * df2['L2_error_p'].iloc[-1]
    ax2.loglog(df2['h'], ref_p2, 'k:', label='O(h¹)', alpha=0.7)
    ax2.set_xlabel('Mesh size ($h$)')
    ax2.set_ylabel('L2 Error')
    ax2.legend(fontsize=11)
    _style_ax(ax2)
    fig2.tight_layout()
    fig2.savefig('kovasznay_convergence_lam2.png', dpi=300)
    plt.close(fig2)

def run_simulation(Re, lam, x, y, mesh_sizes):
    x_min, x_max = x
    y_min, y_max = y

    print(f'\nKovasznay Flow Verification  (Re = {Re},  lambda = {lam:.6f})')
    print(f'{'h':>8}  {'L2(u)':>12}  {'rate_u':>8}  {'L2(p)':>12}  {'rate_p':>8}')
    print('-' * 64)

    errors_u   = []
    errors_p   = []

    # Exact solution functions for BCs and error computation 
    def u_exact(x):
        u = 1.0 - np.exp(lam * x[0]) * np.cos(2.0 * np.pi * x[1])
        v = (lam / (2.0 * np.pi)) * np.exp(lam * x[0]) * np.sin(2.0 * np.pi * x[1])
        return np.stack([u, v])

    def p_exact(x):
        return -0.5 * np.exp(2.0 * lam * x[0]) * np.ones(x.shape[1])

    # Solver setup
    def ksp_setup(ksp):
        ksp.setType(ksp.Type.FGMRES)
        ksp.pc.setType(ksp.pc.Type.LU)
        ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)
        
    for h in mesh_sizes:
        # mesh (tags: 1=left, 2=bottom, 3=right, 4=top) 
        mesh = RectMesh(x_min, y_min, x_max, y_max, h)

        # physics
        nse = SteadyNavierStokes(mesh)
        nse.set_element('Lagrange', 1, 'Lagrange', 1)
        nse.build_function_space()
        nse.set_density(1.0)
        nse.set_dynamic_viscosity(1.0 / Re)
        nse.set_weak_form()
        nse.add_stab()

        V_u = nse.get_function_space('u').collapse()[0]
        V_p = nse.get_function_space('p').collapse()[0]

        # boundary condition functions
        u_bc_fn = dolfinx.fem.Function(V_u)
        u_bc_fn.interpolate(u_exact)

        p_bc_fn = dolfinx.fem.Function(V_p)
        p_bc_fn.interpolate(p_exact)   # constant on the left wall: p(-0.5,y) = const

        # All velocity boundaries Dirichlet; pressure pinned on left wall only.
        # RectMesh tags: 1=left(x=-0.5), 2=bottom(y=-0.5), 3=right(x=1.5), 4=top(y=1.5)
        bc_dict = {
            'u': {
                1: {'type': 'dirichlet', 'value': u_bc_fn},
                2: {'type': 'dirichlet', 'value': u_bc_fn},
                3: {'type': 'dirichlet', 'value': u_bc_fn},
                4: {'type': 'dirichlet', 'value': u_bc_fn},
            },
            'p': {
                1: {'type': 'dirichlet', 'value': p_bc_fn},
            },
        }
        nse.set_bcs(bc_dict)

        problem = NonLinearProblem(nse)
        solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=ksp_setup)

        # solve
        n_iter, converged = solver.solve()
        assert converged, f'Solver did not converge at h={h} (iterations={n_iter})'

        # L2 errors 
        u_sol = nse.get_solution_function().sub(0).collapse()
        p_sol = nse.get_solution_function().sub(1).collapse()

        u_exact_fun = dolfinx.fem.Function(V_u); u_exact_fun.interpolate(u_exact)
        p_exact_fun = dolfinx.fem.Function(V_p); p_exact_fun.interpolate(p_exact)

        comm = mesh.msh.comm

        L2_u = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u_sol - u_exact_fun, u_sol - u_exact_fun) * ufl.dx))
        err_u = np.sqrt(comm.allreduce(L2_u, op=MPI.SUM))

        L2_p = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(p_sol - p_exact_fun, p_sol - p_exact_fun) * ufl.dx))
        err_p = np.sqrt(comm.allreduce(L2_p, op=MPI.SUM))

        errors_u.append(err_u)
        errors_p.append(err_p)

        if len(errors_u) > 1:
            rate_u = np.log(errors_u[-2] / errors_u[-1]) / np.log(2.0)
            rate_p = np.log(errors_p[-2] / errors_p[-1]) / np.log(2.0)
            print(f'{h:>8.4f}  {err_u:>12.4e}  {rate_u:>8.2f}  {err_p:>12.4e}  {rate_p:>8.2f}')
        else:
            print(f'{h:>8.4f}  {err_u:>12.4e}  {'---':>8}  {err_p:>12.4e}  {'---':>8}')

        plot_solution(u_sol, h, lam)

    return errors_u, errors_p

def main():
    Re  = 40.0
    lam_1 = Re / 2.0 + np.sqrt(Re**2 / 4.0 + 4.0 * np.pi**2)
    lam_2 = Re / 2.0 - np.sqrt(Re**2 / 4.0 + 4.0 * np.pi**2)

    errors = {}
    for lam in [lam_1, lam_2]:

        if lam == lam_1:
            # lam_1 is ~41. Narrow domain around x=0 
            x_min, x_max = -0.05, 0.05
            # Domain width is 0.10. We need `h` to be smaller than the width.
            mesh_sizes = [0.1/4, 0.1/8, 0.1/16, 0.1/32, 0.1/64]
        else:
            # lam_2 is ~-0.96. Standard domain.
            x_min, x_max = -0.5, 1.5
            # Domain width is 2.0. We can use coarser meshes than for lam_1
            mesh_sizes = [1/8, 1/16, 1/32, 1/64, 1/128]
        
        y_min, y_max = -0.5, 1.5

        errors_u, errors_p = run_simulation(Re=Re, lam=lam, x=(x_min, x_max), y=(y_min, y_max), mesh_sizes=mesh_sizes)
        errors[lam] = (errors_u, errors_p)

    # Save errors to CSV
    with open('kovasznay_errors.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda', 'h', 'h^2', 'L2_error_u', 'L2_error_p'])
        
        for lam in [lam_1, lam_2]:
            errors_u, errors_p = errors[lam]
            mesh_sizes = [0.1/4, 0.1/8, 0.1/16, 0.1/32] if lam == lam_1 else [1/8, 1/16, 1/32, 1/64]
            
            for h, err_u, err_p in zip(mesh_sizes, errors_u, errors_p):
                writer.writerow([f'{lam:.6f}', f'{h:.6f}', f'{h**2:.6f}', f'{err_u:.6e}', f'{err_p:.6e}'])

    plot_error('kovasznay_errors.csv')
if __name__ == '__main__':
    main()
