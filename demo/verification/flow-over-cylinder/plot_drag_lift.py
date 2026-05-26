import csv
import numpy as np
import matplotlib.pyplot as plt


def _style_ax(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12, width=1.5)
    ax.figure.canvas.draw()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.grid(True, alpha=0.2)


def plot_drag_lift(csv_file='drag_lift.csv'):
    time, cd, cl = [], [], []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            time.append(float(row['time']))
            cd.append(float(row['Cd']))
            cl.append(float(row['Cl']))

    time = np.array(time)
    cd   = np.array(cd)
    cl   = np.array(cl)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time, cd, color='tab:blue', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$C_D$')
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig('drag.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time, cl, color='tab:red', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$C_L$')
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig('lift.png', dpi=300)
    plt.close(fig)

    print('Saved drag.png and lift.png')


if __name__ == '__main__':
    plot_drag_lift()
