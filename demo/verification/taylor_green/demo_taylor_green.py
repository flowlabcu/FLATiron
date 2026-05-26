"""
Taylor-Green Vortex Decay – Temporal Convergence Verification
==============================================================
Exact 2D transient NSE solution with viscous decay (Taylor & Green, 1937).

    u(x,y,t) =  cos(x) * sin(y) * exp(-2*nu*t)
    v(x,y,t) = -sin(x) * cos(y) * exp(-2*nu*t)
    p(x,y,t) = -(cos(2x) + cos(2y)) / 4 * exp(-4*nu*t)

Domain:     [0, 2*pi] x [0, 2*pi]
BCs:        Dirichlet velocity on all 4 walls, updated to exact solution each step.
            No pressure Dirichlet; PSPG pins the constant pressure null space.
IC:         Exact velocity and pressure at t=0.

Elements: P2/P1 Taylor-Hood.
            P2/P1 is inf-sup stable. Stabilization pins the pressure null space and
            avoids setting a pressure BC.

Time integration: Crank-Nicolson (theta=0.5), 2nd-order accurate.

Expected temporal convergence:
            velocity L2 error  ~ O(dt^2)
            pressure L2 error  ~ O(dt)    [single-level pressure: ∇p^{n+1} appears in
                                            momentum rather than ∇p^{n+θ}; velocity still
                                            O(dt^2) by inf-sup decoupling (Heywood &
                                            Rannacher 1990)]

Strategy:
            Fix a fine spatial mesh (P2/P1, h = 2*pi/128) and vary dt over 4 refinement
            levels.  Velocity spatial error O(h^3) ~ 7e-6 << temporal error for all dt.
            Pressure spatial error O(h^2) ~ 2.4e-3; O(dt) temporal convergence is visible
            until dt ~ h^2 ~ 2.4e-3 where the spatial floor is reached.
            A single pressure DOF at the origin is pinned to 0 to remove the constant
            null space. Mean-subtraction in the error norm corrects the offset, so this
            does not contaminate the measured pressure error.

Reference:
            Taylor, G.I. & Green, A.E. (1937). Mechanism of the production of small
            eddies from large ones. Proc. R. Soc. Lond. A, 158, 499-521.
"""

import os
import csv
from matplotlib.ticker import LogLocator
import numpy as np
import dolfinx
import ufl
import pyvista
import matplotlib.pyplot as plt
from mpi4py import MPI

from flatiron_tk.mesh import RectMesh
from flatiron_tk.physics import TransientNavierStokes
from flatiron_tk.solver import NonLinearProblem, NonLinearSolver
from flatiron_tk.info import custom_warning_message

# Global parameters
nu    = 0.5          # kinematic viscosity
T_end = 1.0          # final time
theta = 0.5          # Crank-Nicolson

L = 2.0 * np.pi
h_spatial = L / 128.0  # P2/P1: velocity O(h^3); pressure O(h^2) — 128 drops pressure floor to ~2.4e-3

dt_list = [0.5, 0.1, 0.05, 0.01]

os.makedirs("images", exist_ok=True)


# Exact solution
def u_exact(x, t):
    f = np.exp(-2.0 * nu * t)
    u =  np.cos(x[0]) * np.sin(x[1]) * f
    v = -np.sin(x[0]) * np.cos(x[1]) * f
    return np.stack([u, v])


def make_u_bc(t):
    def _u(x):
        return u_exact(x, t)
    return _u

def p_exact(x, t):
    f = np.exp(-4.0 * nu * t)
    return -(np.cos(2.0 * x[0]) + np.cos(2.0 * x[1])) / 4.0 * f

def make_p_bc(t):
    def _p(x):
        return p_exact(x, t)
    return _p

def ksp_setup(ksp):
    ksp.setType("gmres")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)

def plot_solution(u_sol, dt):
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(u_sol.function_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    block_shape = u_sol.function_space.dofmap.index_map_bs
    u_values = u_sol.x.array.reshape(-1, block_shape)
    u_3d = np.pad(u_values, ((0, 0), (0, 1)), mode="constant")

    grid.point_data["Velocity_3D"] = u_3d
    grid.point_data["Velocity Magnitude"] = np.linalg.norm(u_3d, axis=1)
    grid.set_active_vectors("Velocity_3D")

    plotter = pyvista.Plotter(off_screen=True, window_size=[1000, 800])
    plotter.add_text(f"Taylor-Green dt={dt:.4f}  T={T_end}", font_size=14)
    plotter.add_mesh(grid, scalars="Velocity Magnitude", cmap="viridis", show_scalar_bar=True)

    bounds = grid.bounds
    seed_x = bounds[0] + 0.1 * (bounds[1] - bounds[0])
    streamlines = grid.streamlines(
        vectors="Velocity_3D",
        pointa=(seed_x, bounds[2] + 1e-4, 0.0),
        pointb=(seed_x, bounds[3] - 1e-4, 0.0),
        n_points=50,
        max_steps=5000,
    )
    if streamlines.n_points > 0:
        plotter.add_mesh(streamlines, color="white", line_width=2.0)
    else:
        custom_warning_message(f"Streamlines failed to generate any points for dt={dt:.4f}")

    plotter.view_xy()
    plotter.screenshot(f"images/taylor_green_dt_{dt:.4f}.png")
    plotter.close()

def _style_ax(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12, width=1.5)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.figure.canvas.draw()
    ax.grid(True, alpha=0.2)

def plot_convergence(csv_file):
    dts, errors_u, errors_p = [], [], []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dts.append(float(row["dt"]))
            errors_u.append(float(row["L2_u"]))
            errors_p.append(float(row["L2_p"]))

    dts      = np.array(dts)
    errors_u = np.array(errors_u)
    errors_p = np.array(errors_p)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(dts, errors_u, "o-", color="tab:blue",   linewidth=2, label="Velocity L2 error")
    ax.loglog(dts, errors_p, "s-", color="tab:orange", linewidth=2, label="Pressure L2 error")

    ref_u = (dts / dts[0]) ** 2 * errors_u[0]
    ref_p = (dts / dts[0]) ** 1 * errors_p[0]
    ax.loglog(dts, ref_u, "k--",  alpha=0.7, label=r"$O(\Delta t^2)$")
    ax.loglog(dts, ref_p, "k:",   alpha=0.7, label=r"$O(\Delta t)$")

    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("L2 Error")
    ax.legend(fontsize=11)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig("taylor_green_convergence.png", dpi=300)
    plt.close(fig)

def temporal_convergence():
    errors_u   = []
    errors_p   = []
    rates_u    = []
    rates_p    = []
    all_nsteps = []

    print(f"\nTaylor-Green Vortex Decay  (nu={nu}, T={T_end}, h={h_spatial:.4f})")
    print(f"{'dt':>8}  {'steps':>6}  {'L2(u) at T':>12}  {'rate_u':>8}  {'L2(p) at T':>12}  {'rate_p':>8}")
    print("-" * 72)

    for dt in dt_list:
        n_steps = int(round(T_end / dt))
        all_nsteps.append(n_steps)

        # Mesh (tags: 1=left, 2=bottom, 3=right, 4=top)
        mesh = RectMesh(0.0, 0.0, L, L, h_spatial)

        # Physics: P2/P1 Taylor-Hood
        nse = TransientNavierStokes(mesh)
        nse.set_element("Lagrange", 2, "Lagrange", 1)
        nse.build_function_space()
        nse.set_density(1.0)
        nse.set_dynamic_viscosity(nu)
        nse.set_time_step_size(dt)
        nse.set_midpoint_theta(theta)
        nse.set_weak_form(stab=False)

        V_u = nse.get_function_space("u").collapse()[0]
        V_p = nse.get_function_space("p").collapse()[0]

        u_bc_fn = dolfinx.fem.Function(V_u)
        u_bc_fn.interpolate(make_u_bc(0.0))

        bc_dict = {
            "u": {
                1: {"type": "dirichlet", "value": u_bc_fn},
                2: {"type": "dirichlet", "value": u_bc_fn},
                3: {"type": "dirichlet", "value": u_bc_fn},
                4: {"type": "dirichlet", "value": u_bc_fn},
            },
            "p": {},
        }
        nse.set_bcs(bc_dict)

        # Pin a single pressure DOF at the origin to remove the constant null space.
        # locate_dofs_geometrical with a (sub_space, collapsed_space) tuple returns
        # DOF indices in the PARENT mixed space, which is what the solver needs.
        # One point is enough — mean-subtraction in the error computation corrects the
        # constant offset, so this does not contaminate the L2 pressure error.
        V_p_sub = nse.get_function_space("p")          # sub-space (not collapsed)
        p_dofs  = dolfinx.fem.locate_dofs_geometrical(
            (V_p_sub, V_p),
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
        )
        p_zero = dolfinx.fem.Function(V_p)
        p_zero.x.array[:] = 0.0
        p_pin  = dolfinx.fem.dirichletbc(p_zero, p_dofs, V_p_sub)
        nse.dirichlet_bcs.append(p_pin)

        nse.set_initial_conditions(make_u_bc(0.0), make_p_bc(0.0))
        nse.update_previous_solution()
        nse.set_writer('output', 'pvd')
        nse.write()

        # Build solver
        problem = NonLinearProblem(nse)
        solver  = NonLinearSolver(mesh.msh.comm, problem,
                                  outer_ksp_set_function=ksp_setup,
                                  convergence_criterion='residual')

        # Time loop
        t = 0.0
        for step in range(n_steps):
            t_new = t + dt
            u_bc_fn.interpolate(make_u_bc(t_new))
            _, converged = solver.solve()
            assert converged, (f"Solver did not converge at dt={dt}, step={step+1}/{n_steps}")
            nse.update_previous_solution()
            t = t_new

        # L2 velocity error at T
        comm    = mesh.msh.comm
        u_sol   = nse.solution.sub(0).collapse()
        u_ex_fn = dolfinx.fem.Function(V_u)
        u_ex_fn.interpolate(make_u_bc(T_end))

        err_u = np.sqrt(comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                ufl.inner(u_sol - u_ex_fn, u_sol - u_ex_fn) * ufl.dx
            )),
            op=MPI.SUM,
        ))
        errors_u.append(err_u)

        # L2 pressure error at T (subtract mean to remove any residual null-space offset)
        p_sol = nse.solution.sub(1).collapse()

        vol   = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                dolfinx.fem.Constant(mesh.msh, dolfinx.default_scalar_type(1.0)) * ufl.dx
            )),
            op=MPI.SUM,
        )
        p_mean = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(p_sol * ufl.dx)),
            op=MPI.SUM,
        ) / vol

        p_shifted = dolfinx.fem.Function(V_p)
        p_shifted.x.array[:] = p_sol.x.array - p_mean

        p_ex_fn = dolfinx.fem.Function(V_p)
        p_ex_fn.interpolate(make_p_bc(T_end))

        err_p = np.sqrt(comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                ufl.inner(p_shifted - p_ex_fn, p_shifted - p_ex_fn) * ufl.dx
            )),
            op=MPI.SUM,
        ))
        errors_p.append(err_p)

        if len(errors_u) > 1:
            ru = np.log(errors_u[-2] / errors_u[-1]) / np.log(2.0)
            rp = np.log(errors_p[-2] / errors_p[-1]) / np.log(2.0)
            rates_u.append(ru)
            rates_p.append(rp)
            print(f"{dt:>8.4f}  {n_steps:>6}  {err_u:>12.4e}  {ru:>8.2f}  {err_p:>12.4e}  {rp:>8.2f}")
        else:
            rates_u.append(float("nan"))
            rates_p.append(float("nan"))
            print(f"{dt:>8.4f}  {n_steps:>6}  {err_u:>12.4e}  {'---':>8}  {err_p:>12.4e}  {'---':>8}")

        plot_solution(u_sol, dt)

    # Pressure floor: P1 gives O(h^2); temporal convergence saturates below this level
    p_spatial_floor = h_spatial ** 2
    print()
    print("Expected: velocity L2 error ~ O(dt^2)  [Crank-Nicolson, theta=0.5]")
    print("Expected: pressure L2 error ~ O(dt)    [single-level pressure; spatial floor "
          f"~O(h^2) = {p_spatial_floor:.2e}]")

    csv_file = "taylor_green_errors.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dt", "n_steps", "L2_u", "rate_u", "L2_p", "rate_p"])
        for dt, ns, eu, ru, ep, rp in zip(dt_list, all_nsteps, errors_u, rates_u, errors_p, rates_p):
            ru_str = f"{ru:.4f}" if not np.isnan(ru) else "---"
            rp_str = f"{rp:.4f}" if not np.isnan(rp) else "---"
            writer.writerow([f"{dt:.6f}", ns, f"{eu:.6e}", ru_str, f"{ep:.6e}", rp_str])

    plot_convergence(csv_file)
    print(f"\nResults saved to {csv_file} and taylor_green_convergence.png")
    print("Solution images saved to images/taylor_green_dt_*.png")

if __name__ == '__main__':
    temporal_convergence()
