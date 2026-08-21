"""
Plot flatiron_tk vs bare-dolfinx per-stage wall-clock time as a function of
problem size (DOFs/rank), holding rank count fixed -- the raw-time
complement of plot_dof_efficiency_3d.py (which plots t[1]/t[N] efficiency
instead).

Reads flatiron_dof_sweep_3d.csv / dolfinx_dof_sweep_3d.csv (produced by
flatiron_dof_sweep_3d.py / dolfinx_dof_sweep_3d.py) and, for every rank
count x stage found in the CSVs, writes one PNG into plots/:

    plots/dof_sweep_rank{N}_{stage}_time.png

Usage:
    python3 plot_dof_sweep_3d.py
"""
import csv
import os

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

COLOR_FLATIRON = 'black'
COLOR_DOLFINX = 'red'

STAGES = ['mesh_s', 'setup_s', 'assemble_s', 'pc_setup_s', 'solve_s', 'total_s']
STAGE_LABELS = {
    'mesh_s': 'mesh construction time (s)',
    'setup_s': 'function space + weak form + BC setup time (s)',
    'assemble_s': 'assembly time (s)',
    'pc_setup_s': 'PC (AMG hierarchy) setup time (s)',
    'solve_s': 'KSP solve time (s)',
    'total_s': 'total time (s)',
}


def _read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in ('num_ranks', 'target_dofs_per_rank', 'ne', 'dofs_global', 'ksp_iters'):
                    parsed[k] = int(v)
                else:
                    parsed[k] = float(v)
            parsed['total_s'] = (parsed['mesh_s'] + parsed['setup_s']
                                  + parsed['assemble_s'] + parsed['pc_setup_s']
                                  + parsed['solve_s'])
            rows.append(parsed)
    return rows


def _style_axes(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=16, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='both', labelsize=14, width=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.grid(True, alpha=0.2)


def _rows_for_rank(rows, rank):
    r = [row for row in rows if row['num_ranks'] == rank]
    r.sort(key=lambda row: row['target_dofs_per_rank'])
    return r


def make_plot(flatiron_rows, dolfinx_rows, rank, stage):
    dofs = [r['dofs_per_rank'] for r in flatiron_rows]
    flatiron_vals = [r[stage] for r in flatiron_rows]
    dolfinx_vals = [r[stage] for r in dolfinx_rows]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    _style_axes(ax)

    ax.plot(dofs, flatiron_vals, color=COLOR_FLATIRON, linewidth=2, marker='o',
             markersize=6, markerfacecolor=COLOR_FLATIRON, markeredgecolor='white',
             markeredgewidth=1, label='flatiron_tk', zorder=3)
    ax.plot(dofs, dolfinx_vals, color=COLOR_DOLFINX, linewidth=2, marker='o',
             markersize=6, markerfacecolor=COLOR_DOLFINX, markeredgecolor='white',
             markeredgewidth=1, label='dolfinx (bare)', zorder=3)

    ax.set_xlabel(f'DOFs / rank ({rank} rank{"s" if rank != 1 else ""})', color='black')
    ax.set_ylabel(STAGE_LABELS[stage], color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, labelcolor='black', fontsize=12, loc='best')

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f'dof_sweep_rank{rank}_{stage.rstrip("_s")}_time.png')
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)
    return out_path


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    flatiron_rows = _read_csv(os.path.join(SCRIPT_DIR, 'flatiron_dof_sweep_3d.csv'))
    dolfinx_rows = _read_csv(os.path.join(SCRIPT_DIR, 'dolfinx_dof_sweep_3d.csv'))

    ranks = sorted(set(r['num_ranks'] for r in flatiron_rows)
                    & set(r['num_ranks'] for r in dolfinx_rows))

    written = []
    for rank in ranks:
        f_rows = _rows_for_rank(flatiron_rows, rank)
        d_rows = _rows_for_rank(dolfinx_rows, rank)

        f_targets = [r['target_dofs_per_rank'] for r in f_rows]
        d_targets = [r['target_dofs_per_rank'] for r in d_rows]
        if f_targets != d_targets:
            print(f'Warning: rank={rank} target_dofs_per_rank values differ between '
                  f'flatiron ({f_targets}) and dolfinx ({d_targets}); skipping.')
            continue

        for stage in STAGES:
            out_path = make_plot(f_rows, d_rows, rank, stage)
            written.append(out_path)

    print(f'Wrote {len(written)} plots to {PLOTS_DIR}')
    for p in written:
        print(f'  {p}')


if __name__ == '__main__':
    main()
