"""
Plot temporal convergence from taylor_green_errors.csv.
Run from the taylor_green/ directory:
    python plot_taylor_green_convergence.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

plt.rcParams['mathtext.fontset'] = 'stix'


def _log_tick_format(value, pos):
    exponent = int(np.round(np.log10(value)))
    return r'$\mathbf{1 \times 10^{%d}}$' % exponent


def _style_ax(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=16, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='both', labelsize=14, width=1.5)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(FuncFormatter(_log_tick_format))
    ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_format))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.figure.canvas.draw()
    ax.grid(True, alpha=0.2)


def plot_convergence(csv_file):
    dts, errors_u, errors_p = [], [], []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dts.append(float(row['dt']))
            errors_u.append(float(row['L2_u']))
            errors_p.append(float(row['L2_p']))

    dts      = np.array(dts)
    errors_u = np.array(errors_u)
    errors_p = np.array(errors_p)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.loglog(dts, errors_u, 'o-', color='limegreen',   linewidth=2, label=r'$\mathbf{u}$ error')
    ax.loglog(dts, errors_p, 's-', color='turquoise', linewidth=2, label=r'$p$ error')

    ref_u = (dts / dts[0]) ** 2 * errors_u[0]
    ref_p = (dts / dts[0]) ** 1 * errors_p[0]
    ax.loglog(dts, ref_u, 'k--', alpha=0.7, label=r'$O(\boldsymbol{\Delta t}^2)$')
    ax.loglog(dts, ref_p, 'k:',  alpha=0.7, label=r'$O(\boldsymbol{\Delta t})$')

    ax.set_xlabel(r'Time Step Size')
    ax.set_ylabel(r'L2 Error')
    ax.legend(fontsize=11)
    _style_ax(ax)
    fig.tight_layout()

    out = 'taylor_green_convergence.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    plot_convergence('taylor_green_errors.csv')
