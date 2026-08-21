"""
Plot spatial convergence from kovasznay_errors.csv.
Run from the kovasznay/ directory:
    python plot_kovasznay_convergence.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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



def plot_error(file):
    df = pd.read_csv(file)

    for lam_sign, suffix, title_lam in [(1, 'lam1', r'$\lambda_1$'), (-1, 'lam2', r'$\lambda_2$')]:
        sub = df[df['lambda'] * lam_sign > 0].sort_values('h')


        fig, ax = plt.subplots(figsize=(6, 5))
        

        ax.loglog(sub['h'], sub['L2_error_u'], 'o-',
                  label=r'$\mathbf{u}$ error', color='tab:blue', linewidth=2)
        ax.loglog(sub['h'], sub['L2_error_p'], 's-',
                  label=r'$p$ error', color='tab:red', linewidth=2)

        ref_u = (sub['h'] / sub['h'].iloc[-1]) ** 2 * sub['L2_error_u'].iloc[-1]
        ax.loglog(sub['h'], ref_u, 'k--', alpha=0.7, label=r'$O(\mathbf{h}^2)$')

        ref_p = (sub['h'] / sub['h'].iloc[-1]) ** 1 * sub['L2_error_p'].iloc[-1]
        ax.loglog(sub['h'], ref_p, 'k:', alpha=0.7, label=r'$O(\mathbf{h})$')

        ax.set_xlabel('Mesh size', fontsize=16, fontweight='bold')
        ax.set_ylabel('L2 Error', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        _style_ax(ax)
        fig.tight_layout()

        out = f'kovasznay_convergence_{suffix}.png'
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f'Saved {out}')


if __name__ == '__main__':
    plot_error('kovasznay_errors.csv')
