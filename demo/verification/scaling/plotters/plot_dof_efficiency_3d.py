"""
Plot solve efficiency (t[1]/t[N]) as a function of problem granularity
(DOFs/rank), holding rank count fixed -- the complement of the
rank-count sweep in plot_reference_comparison_3d.py.

Requires flatiron_dof_sweep_3d.csv / dolfinx_dof_sweep_3d.csv to contain
rows for both num_ranks=1 (baseline) and one other num_ranks value, at the
same target_dofs_per_rank values (run flatiron_dof_sweep_3d.py /
dolfinx_dof_sweep_3d.py at both rank counts first).

Usage:
    python3 plot_dof_efficiency_3d.py [--ranks N]
"""
import argparse
import csv
import os
import statistics

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

COLOR_FLATIRON = 'black'
COLOR_DOLFINX = 'red'


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


def _median_total_by_target(rows, target_rank):
    by_target = {}
    for r in rows:
        if r['num_ranks'] != target_rank:
            continue
        by_target.setdefault(r['target_dofs_per_rank'], []).append(r['pc_setup_s'] + r['solve_s'])
    return {t: statistics.median(v) for t, v in by_target.items()}


def _efficiency_by_target(rows, high_rank):
    t1 = _median_total_by_target(rows, 1)
    tn = _median_total_by_target(rows, high_rank)
    targets = sorted(set(t1) & set(tn))
    eff = [t1[t] / tn[t] for t in targets]
    return targets, eff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranks', type=int, default=None,
                         help='Rank count to compare against the rank=1 baseline '
                              '(default: highest num_ranks found in the CSVs, excluding 1).')
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    flatiron_rows = _read_csv(os.path.join(SCRIPT_DIR, 'flatiron_dof_sweep_3d.csv'))
    dolfinx_rows = _read_csv(os.path.join(SCRIPT_DIR, 'dolfinx_dof_sweep_3d.csv'))

    available_ranks = sorted({r['num_ranks'] for r in flatiron_rows + dolfinx_rows} - {1})
    high_rank = args.ranks or (available_ranks[-1] if available_ranks else None)
    if high_rank is None:
        raise SystemExit('No non-1 num_ranks found in the CSVs -- run the sweep at rank=1 and rank=N first.')

    flatiron_targets, flatiron_eff = _efficiency_by_target(flatiron_rows, high_rank)
    dolfinx_targets, dolfinx_eff = _efficiency_by_target(dolfinx_rows, high_rank)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    _style_axes(ax)
    ax.axhline(1.0, color='lightgray', linewidth=1, linestyle='--', zorder=1)

    ax.plot(flatiron_targets, flatiron_eff, color=COLOR_FLATIRON, linewidth=2,
             marker='o', markersize=6, markerfacecolor=COLOR_FLATIRON,
             markeredgecolor='white', markeredgewidth=1, label='flatiron_tk (3D)', zorder=3)
    ax.plot(dolfinx_targets, dolfinx_eff, color=COLOR_DOLFINX, linewidth=2,
             marker='o', markersize=6, markerfacecolor=COLOR_DOLFINX,
             markeredgecolor='white', markeredgewidth=1, label='dolfinx (bare, 3D)', zorder=3)

    ax.set_xlabel('target DOFs / rank', color='black')
    ax.set_ylabel(f'PC setup + solve efficiency (t[1]/t[{high_rank}])', color='black')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, labelcolor='black', fontsize=12, loc='lower right')

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f'dof_efficiency_3d_rank{high_rank}.png')
    fig.savefig(out_path, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')

    print()
    print(f'Efficiency (t[1]/t[{high_rank}]) by target DOFs/rank:')
    print(f'{"target dofs/rank":>18} {"flatiron_tk":>12} {"dolfinx":>10}')
    all_targets = sorted(set(flatiron_targets) | set(dolfinx_targets))
    for t in all_targets:
        f = dict(zip(flatiron_targets, flatiron_eff)).get(t)
        d = dict(zip(dolfinx_targets, dolfinx_eff)).get(t)
        fmt = lambda x: f'{x:.1%}' if x is not None else '--'
        print(f'{t:>18,} {fmt(f):>12} {fmt(d):>10}')


if __name__ == '__main__':
    main()
