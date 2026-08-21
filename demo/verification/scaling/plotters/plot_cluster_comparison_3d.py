"""
Three-way weak-scaling efficiency comparison: local desktop (8 cores,
shared memory bus) vs. a single real HPC node (1 node / 16 tasks,
Alpine/CURC cluster) vs. the FEniCS/performance-test reference (their own
published numbers, likely multi-node). All three use the same 3D unit-cube
/ tetrahedron problem and reference-matched KSP/AMG settings
(strong_threshold=0.7, agg_nl=4, agg_num_paths=2, ghost_mode=none).

Desktop and cluster data are each median-of-5-trials per rank count.

Usage:
    python3 plot_cluster_comparison_3d.py
"""
import csv
import os
import statistics

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

COLOR_DESKTOP = 'black'
COLOR_CLUSTER = 'red'
COLOR_REFERENCE = 'green'

REFERENCE_RANKS = [1, 2, 4, 8, 16]
REFERENCE_TIME_S = [7.35, 8.38, 9.57, 9.8, 10.95]

# Local desktop has 8 physical cores; the 8-rank point there is
# contaminated by CPU contention (established repeatedly earlier).
DESKTOP_CONTAMINATED_RANKS = {8}


def _read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in ('num_ranks', 'ne', 'dofs_global', 'ksp_iters'):
                    parsed[k] = int(v)
                else:
                    parsed[k] = float(v)
            rows.append(parsed)
    return rows


def _median_total_by_rank(rows):
    by_rank = {}
    for r in rows:
        by_rank.setdefault(r['num_ranks'], []).append(r['pc_setup_s'] + r['solve_s'])
    ranks = sorted(by_rank)
    times = [statistics.median(by_rank[r]) for r in ranks]
    return ranks, times


def _efficiency(times):
    t1 = times[0]
    return [t1 / t for t in times]


def _style_axes(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=16, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='both', labelsize=14, width=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.grid(True, alpha=0.2)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Desktop: combine flatiron + dolfinx (they track each other closely;
    # average the two implementations' medians to get one desktop curve).
    desktop_rows = (_read_csv(os.path.join(SCRIPT_DIR, 'flatiron_weak_scaling_3d.csv'))
                     + _read_csv(os.path.join(SCRIPT_DIR, 'dolfinx_weak_scaling_3d.csv')))
    desktop_ranks, desktop_totals = _median_total_by_rank(desktop_rows)
    desktop_eff = _efficiency(desktop_totals)

    cluster_rows = _read_csv(os.path.join(SCRIPT_DIR, 'cluster_data', 'cluster_weak_scaling_3d.csv'))
    cluster_ranks, cluster_totals = _median_total_by_rank(cluster_rows)
    cluster_eff = _efficiency(cluster_totals)

    reference_eff = _efficiency(REFERENCE_TIME_S)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    _style_axes(ax)
    ax.axhline(1.0, color='lightgray', linewidth=1, linestyle='--', zorder=1)

    def plot_series(ranks, values, color, label, contaminated=None):
        contaminated = contaminated or set()
        clean = [(r, v) for r, v in zip(ranks, values) if r not in contaminated]
        flagged = [(r, v) for r, v in zip(ranks, values) if r in contaminated]
        if clean:
            cr, cv = zip(*clean)
            ax.plot(cr, cv, color=color, linewidth=2, marker='o', markersize=6,
                     markerfacecolor=color, markeredgecolor='white',
                     markeredgewidth=1, label=label, zorder=3)
        if clean and flagged:
            ax.plot([clean[-1][0], flagged[0][0]], [clean[-1][1], flagged[0][1]],
                     color=color, linewidth=1.5, linestyle=':', zorder=2)
        if flagged:
            fr, fv = zip(*flagged)
            ax.plot(fr, fv, color=color, linewidth=0, marker='o', markersize=7,
                     markerfacecolor='white', markeredgecolor=color,
                     markeredgewidth=1.5, zorder=3)

    plot_series(desktop_ranks, desktop_eff, COLOR_DESKTOP, 'local desktop (8 cores)',
                contaminated=DESKTOP_CONTAMINATED_RANKS)
    plot_series(cluster_ranks, cluster_eff, COLOR_CLUSTER, 'cluster (1 node, 16 tasks)')
    plot_series(REFERENCE_RANKS, reference_eff, COLOR_REFERENCE, 'FEniCS/performance-test (reference)')

    ax.set_xlabel('MPI ranks (weak scaling)', color='black')
    ax.set_ylabel('PC setup + solve efficiency (t[1]/t[n])', color='black')
    ax.set_xscale('log', base=2)
    all_ranks = sorted(set(desktop_ranks) | set(cluster_ranks) | set(REFERENCE_RANKS))
    ax.set_xticks(all_ranks)
    ax.set_xticklabels([str(r) for r in all_ranks])
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, labelcolor='black', fontsize=12, loc='lower left')

    fig.text(0.5, -0.02,
              'hollow marker: desktop 8-rank point contaminated by local CPU contention (8 cores) -- not a real scaling signal',
              ha='center', fontsize=8, color='lightgray')

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'weak_efficiency_desktop_vs_cluster_vs_reference_3d.png')
    fig.savefig(out_path, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')

    print()
    print('Efficiency ratios (t[1]/t[n]), median of 5 trials (desktop, cluster):')
    print(f'{"ranks":>6} {"desktop":>10} {"cluster":>10} {"reference":>10}')
    for r in all_ranks:
        d = dict(zip(desktop_ranks, desktop_eff)).get(r)
        c = dict(zip(cluster_ranks, cluster_eff)).get(r)
        ref = dict(zip(REFERENCE_RANKS, reference_eff)).get(r)
        fmt = lambda x: f'{x:.1%}' if x is not None else '--'
        print(f'{r:>6} {fmt(d):>10} {fmt(c):>10} {fmt(ref):>10}')


if __name__ == '__main__':
    main()
