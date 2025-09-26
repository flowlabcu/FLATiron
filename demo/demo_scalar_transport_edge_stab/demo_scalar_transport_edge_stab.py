import dolfinx
import flatiron_tk
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
from flatiron_tk.physics import TransientScalarTransport
from flatiron_tk.mesh import RectMesh
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver
from flatiron_tk.io import bp_mod

def solve_pure_advection(num_elements, gamma, SUPG=True, fixed_cfl=True):
    # Create mesh 
    L = 1 
    h = L/num_elements
    mesh = RectMesh(0,0, L, L/10, h)

    # Advection speed beta and diffusivity epsilon
    # If fixed_cfl, we fix the cfl to 0.1, i.e., dt/h=0.1 for unit velocity
    if fixed_cfl:
        dt = 0.1*h
    else:
        dt = 1e-3 

    theta = 0.5
    beta = flatiron_tk.constant(mesh, (1.0, 0.0))

    stp = TransientScalarTransport(mesh, dt, theta)
    stp.set_tag('c')
    stp.set_element('CG', 1)
    stp.build_function_space()

    # Diffusivity (here set as a constant)
    D = 0.0
    stp.set_diffusivity(D, D)

    # Velocity 
    stp.set_advection_velocity(beta, beta)

    # Reaction 
    R = 0.0
    stp.set_reaction(R, R)

    # Set the weak form
    stp.set_weak_form()

    # Standard SUPG stabilization
    if SUPG:
        stp.add_stab()

    # Add edge stab if gamma in > 1e-12
    if abs(gamma) > 1e-12:
        stp.add_edge_stab(gamma)

    bc_dict = {1: {'type':'dirichlet', 'value': flatiron_tk.constant(mesh, 1.0)}}
    stp.set_bcs(bc_dict)

    # Set problem and solver
    problem = NonLinearProblem(stp)
    solver = NonLinearSolver(mesh.msh.comm, problem)

    # Set output directory based on parameters
    output_dir = f'output_{num_elements}_{gamma:.5f}'
    if SUPG:
        output_dir += '_SUPG'
    
    if fixed_cfl:
        output_dir += '_fixed_cfl'
    
    # Set output writer
    stp.set_writer(output_dir, 'bp')

    # Solve
    num_time_steps = int(0.5/dt) + 1
    for i in range(num_time_steps):
        solver.solve()

        stp.update_previous_solution()

        if mesh.msh.comm.rank == 0:
            print(f'h={h:.4f}, time step {i}/{num_time_steps} complete.')
    sys.stdout.flush()
    
    stp.write(time_stamp=0.5)

def read_data(ne, gamma, fixed_cfl):
    if fixed_cfl: 
        file_name = f'output_{ne}_{gamma:.5f}_SUPG_fixed_cfl/c.bp'
    else:
        file_name = f'output_{ne}_{gamma:.5f}_SUPG/c.bp'

    c = bp_mod.bp_read_function(file_name, time_id=-1, name='c', 
                                element_family='CG', element_degree=1, element_shape='scalar')

    L = 1.0
    y_mid = 0.5 * (L / 10)
    x = np.linspace(0, 1, ne + 1)
    cmid = np.zeros(ne + 1)

    mesh = c.function_space.mesh
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

    # Evaluate the solution at the midline y=0.5
    for i, xi in enumerate(x):
        pt = np.array([xi, y_mid, 0.0], dtype=np.float64)
        cells = dolfinx.geometry.compute_collisions_points(tree, pt)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cells, np.array([pt], dtype=np.float64))
        cell_candidates = colliding_cells.links(0)

        if len(cell_candidates) > 0:
            cell_index = cell_candidates[0]
            cmid[i] = c.eval(pt, cell_index)[0]
        else:
            raise RuntimeError(f"Point {pt} not in mesh")

    return x, cmid

def plot_data():
    matplotlib.rcParams.update({'font.size': 12, 'font.weight': 'bold'})

    xx = np.linspace(0, 1, 1000)
    H = np.heaviside(0.5 - xx, 1)

    col = {100: [1,0,0],
       200: [2/3,0.5/3,1/3],
       400: [1/3,1/3,2/3],
       800: [0,0.5,1]}

    a = 2
    fig, axs = plt.subplots(ncols=2, figsize=(a*6.777,a*2.475))

    for ne in [100, 200, 400, 800]:
        mod = ne//100
        if mod == 1:
            label = "$h=10^{-2}$"
        else:
            label = "$h=\\frac{1}{%d}10^{-2}$"%mod
        x, c = read_data(ne, 0, fixed_cfl=True)
        axs[0].plot(x, c, '-', label=label, color=col[ne])

        x, c = read_data(ne, 0.01, fixed_cfl=True)
        axs[1].plot(x, c, '-', label=label, color=col[ne])

    for ax in axs:
        ax.plot(xx, H, 'k--', label='H(0.5-x)')
        ax.set_xlim([0.4, 0.6])
        ax.legend(ncols=1, fontsize=10)
        ax.grid(True)
        ax.set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])
        ax.set_xlabel('x')
        ax.set_ylim(top=1.2)
        ax.set_ylabel('$c_{mid}$')
    plt.tight_layout()
    plt.savefig('fig1_b1_b2.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    solve_pure_advection(num_elements=100, gamma=0.0, fixed_cfl=True)
    solve_pure_advection(num_elements=100, gamma=0.0, fixed_cfl=False)
    solve_pure_advection(num_elements=200, gamma=0.0, fixed_cfl=True)
    solve_pure_advection(num_elements=200, gamma=0.0, fixed_cfl=False)
    solve_pure_advection(num_elements=400, gamma=0.0, fixed_cfl=True)
    solve_pure_advection(num_elements=400, gamma=0.0, fixed_cfl=False)
    solve_pure_advection(num_elements=800, gamma=0.0, fixed_cfl=True)
    solve_pure_advection(num_elements=800, gamma=0.0, fixed_cfl=False)

    solve_pure_advection(num_elements=100, gamma=1e-2, fixed_cfl=True)
    solve_pure_advection(num_elements=100, gamma=1e-2, fixed_cfl=False)
    solve_pure_advection(num_elements=200, gamma=1e-2, fixed_cfl=True)
    solve_pure_advection(num_elements=200, gamma=1e-2, fixed_cfl=False)
    solve_pure_advection(num_elements=400, gamma=1e-2, fixed_cfl=True)
    solve_pure_advection(num_elements=400, gamma=1e-2, fixed_cfl=False)
    solve_pure_advection(num_elements=800, gamma=1e-2, fixed_cfl=True)
    solve_pure_advection(num_elements=800, gamma=1e-2, fixed_cfl=False)

    plot_data()