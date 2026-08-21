"""
Compare flatiron_tk vs bare-dolfinx Poisson scaling results -- 3D version,
using flatiron_scaling_3d.py / dolfinx_scaling_3d.py results (unit cube,
tetrahedra) instead of the 2D approximation.

Reads the four CSVs produced by flatiron_scaling_3d.py / dolfinx_scaling_3d.py
(one row per rank count per scaling type) and, for every stage x scaling
type x metric combination, writes one PNG into plots/:

    stages:       mesh, setup, assemble, pc_setup, solve, total
    scaling types: strong, weak
    metrics:      time (raw wall-clock seconds), efficiency
                  -- weak:   t[1]/t[n]        (ideal = 100% at every n)
                  -- strong: (t[1]/t[n]) / n  (ideal = 100% at every n;
                             this is speedup normalized by rank count, NOT
                             the raw ratio -- ideal strong scaling means
                             t(n)=t(1)/n, so the raw ratio t[1]/t[n] should
                             equal n, not 1, at perfect efficiency)

6 stages x 2 scaling types x 2 metrics = 24 PNGs, e.g.
    plots/weak_solve_time_3d.png
    plots/weak_solve_efficiency_3d.png
    plots/strong_assemble_time_3d.png

Usage:
    python3 plot_scaling_comparison_3d.py
"""
import csv
import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR = os.path.join(DATA_DIR, 'plots')

# Categorical palette (dataviz skill reference palette, slots 1/2 -- blue vs
# orange, validated for CVD separation).
COLOR_FLATIRON = 'black'
COLOR_DOLFINX = 'red'
COLOR_REFERENCE = 'green'

REFERENCE_RANKS = [1, 2, 4, 8, 16] # Weak scaling reference data from FEniCS/performance-test dashboard
REFERENCE_TIME_S = [7.35, 8.38, 9.57, 9.8, 10.95]


STAGES = ['mesh_s', 'setup_s', 'assemble_s', 'pc_setup_s', 'solve_s', 'total_s']
STAGE_LABELS = {
    'mesh_s': 'mesh construction time (s)',
    'setup_s': 'function space + weak form + BC setup time (s)',
    'assemble_s': 'assembly time (s)',
    'pc_setup_s': 'PC (AMG hierarchy) setup time (s)',
    'solve_s': 'KSP solve time (s)',
    'total_s': 'Wall Time (s)',
}
SCALING_TYPES = ['strong', 'weak']


def _read_csv(path):
    """Read a scaling CSV and collapse repeated trials (same num_ranks) down
    to one row per rank count by taking the median of each metric."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k == 'num_ranks' or k == 'ne' or k == 'dofs_global' or k == 'ksp_iters':
                    parsed[k] = int(v)
                else:
                    parsed[k] = float(v)
            parsed['total_s'] = (parsed['mesh_s'] + parsed['setup_s']
                                  + parsed['assemble_s'] + parsed['pc_setup_s']
                                  + parsed['solve_s'])
            rows.append(parsed)

    trials_by_rank = defaultdict(list)
    for row in rows:
        trials_by_rank[row['num_ranks']].append(row)

    medians = []
    for num_ranks, trials in trials_by_rank.items():
        keys = trials[0].keys()
        medians.append({k: statistics.median(t[k] for t in trials) for k in keys})
    medians.sort(key=lambda r: r['num_ranks'])
    return medians


def _style_axes(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=16, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='both', labelsize=14, width=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.grid(True, alpha=0.2)

def _plot_series(ax, ranks, values, color, label):
    ax.plot(ranks, values, color=color, linewidth=2, marker='o',
             markersize=6, markerfacecolor=color, markeredgecolor='white',
             markeredgewidth=1, label=label, zorder=3)


def make_plot(flatiron_rows, dolfinx_rows, stage, scaling_type, metric, do_reference=False):
    ranks = [r['num_ranks'] for r in flatiron_rows]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    _style_axes(ax)

    if metric == 'time':
        flatiron_vals = [r[stage] for r in flatiron_rows]
        dolfinx_vals = [r[stage] for r in dolfinx_rows]
        ax.set_ylabel(STAGE_LABELS[stage], color='black')
    else:
        flatiron_t1 = flatiron_rows[0][stage]
        dolfinx_t1 = dolfinx_rows[0][stage]
        base_rank = flatiron_rows[0]['num_ranks']
        if scaling_type == 'strong':
            # E(n) = (t[1]/t[n]) / n 
            flatiron_vals = [(flatiron_t1 / r[stage]) / (r['num_ranks'] / base_rank)
                              for r in flatiron_rows]
            dolfinx_vals = [(dolfinx_t1 / r[stage]) / (r['num_ranks'] / base_rank)
                             for r in dolfinx_rows]
            label = 'Efficiency'
        else:
            flatiron_vals = [flatiron_t1 / r[stage] for r in flatiron_rows]
            dolfinx_vals = [dolfinx_t1 / r[stage] for r in dolfinx_rows]
            label = 'Efficiency'
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(f'{label}', color='black')
        ax.axhline(1.0, color='lightgray', linewidth=1, linestyle='--', zorder=1)

    _plot_series(ax, ranks, flatiron_vals, COLOR_FLATIRON, 'flatiron_tk')
    _plot_series(ax, ranks, dolfinx_vals, COLOR_DOLFINX, 'dolfinx (bare)')

    all_ranks = ranks
    if do_reference:
        reference_t1 = REFERENCE_TIME_S[0]
        reference_vals = [reference_t1 / t for t in REFERENCE_TIME_S]
        _plot_series(ax, REFERENCE_RANKS, reference_vals, COLOR_REFERENCE,
                     'FEniCS/performance-test')
        all_ranks = sorted(set(ranks) | set(REFERENCE_RANKS))

    if metric == 'time' and scaling_type == 'strong':
        ax.set_ylim(bottom=0)

    ax.set_xlabel(f'MPI ranks', color='black')
    ax.set_xscale('log', base=2)
    ax.set_xticks(all_ranks)
    ax.set_xticklabels([str(r) for r in all_ranks])
    ax.legend(frameon=False, labelcolor='black', fontsize=12, loc='best')

    fig.tight_layout()
    suffix = '_vs_reference' if do_reference else ''
    out_path = os.path.join(PLOTS_DIR, f'{scaling_type}_{stage.rstrip("_s")}_{metric}{suffix}_3d.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    written = []
    for scaling_type in SCALING_TYPES:
        flatiron_csv = os.path.join(DATA_DIR, f'flatiron_{scaling_type}_scaling_3d.csv')
        dolfinx_csv = os.path.join(DATA_DIR, f'dolfinx_{scaling_type}_scaling_3d.csv')

        if not (os.path.exists(flatiron_csv) and os.path.exists(dolfinx_csv)):
            print(f'Skipping {scaling_type} scaling: missing CSV(s). '
                  f'Expected {flatiron_csv} and {dolfinx_csv}.')
            continue

        flatiron_rows = _read_csv(flatiron_csv)
        dolfinx_rows = _read_csv(dolfinx_csv)

        flatiron_ranks = [r['num_ranks'] for r in flatiron_rows]
        dolfinx_ranks = [r['num_ranks'] for r in dolfinx_rows]
        if flatiron_ranks != dolfinx_ranks:
            print(f'Warning: {scaling_type} scaling rank counts differ between '
                  f'flatiron ({flatiron_ranks}) and dolfinx ({dolfinx_ranks}); '
                  'plotting the overlap only is not implemented -- fix the CSVs.')
            continue

        for stage in STAGES:
            for metric in ('time', 'efficiency'):
                out_path = make_plot(flatiron_rows, dolfinx_rows, stage, scaling_type, metric)
                written.append(out_path)

        if scaling_type == 'weak':
            # Reference dashboard data is only available as a single wall-clock
            # number per rank count, so the reference overlay only makes sense
            # against total_s efficiency, not the per-stage breakdown.
            out_path = make_plot(flatiron_rows, dolfinx_rows, 'total_s', 'weak', 'efficiency',
                                  do_reference=True)
            written.append(out_path)

    print(f'Wrote {len(written)} plots to {PLOTS_DIR}')
    for p in written:
        print(f'  {p}')


if __name__ == '__main__':
    main()
