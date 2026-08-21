import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def _style_ax(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='both', labelsize=14, width=1.5)
    ax.figure.canvas.draw()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.grid(True, alpha=0.2)


def _load_source(csv_file):
    x, y = [], []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row['x']))
            y.append(float(row['y']))
    return np.array(x), np.array(y)


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

    cl_src_t, cl_src = _load_source('cl-source.csv')
    cd_g2_t, cd_g2 = _load_source('cd-source-g2.csv')
    cd_g4_t, cd_g4 = _load_source('cd-source-g4.csv')

    # Phase-align: shift simulation time so first Cd peak matches first source Cd peak
    sim_peaks, _ = find_peaks(cd)
    src_peaks, _ = find_peaks(cd_g4)
    t_shift = cd_g4_t[src_peaks[0]] - time[sim_peaks[0]]
    time = time + t_shift

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cd_g2_t, cd_g2, color='black', linewidth=1.5, linestyle=':', label='Ferziger - Coarse Grid')
    ax.plot(cd_g4_t, cd_g4, color='black', linewidth=1.5, linestyle='--', label='Ferziger - Fine Grid')
    ax.plot(time, cd, color='blue', linewidth=2, label='FLATiron')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\mathbf{C_D}$')
    ax.set_xlim(0, 0.30)
    ax.set_ylim(3.15, 3.25)
    ax.legend(fontsize=12)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig('drag.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cl_src_t, cl_src, color='black', linewidth=1.5, linestyle='--', label='Ferziger')
    ax.plot(time, cl, color='blue', linewidth=2, label='FLATiron')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\mathbf{C_L}$')
    ax.set_xlim(0, 0.30)
    ax.legend(fontsize=12)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig('lift.png', dpi=300)
    plt.close(fig)

    # Compute means only over the time window where all datasets overlap
    cd_t0 = max(time.min(), cd_g2_t.min(), cd_g4_t.min())
    cd_t1 = min(time.max(), cd_g2_t.max(), cd_g4_t.max())

    sim_cd_mean = cd[(time >= cd_t0) & (time <= cd_t1)].mean()
    g2_mean     = cd_g2[(cd_g2_t >= cd_t0) & (cd_g2_t <= cd_t1)].mean()
    g4_mean     = cd_g4[(cd_g4_t >= cd_t0) & (cd_g4_t <= cd_t1)].mean()

    print('Saved drag.png and lift.png')
    print()
    print(f'Mean Cd (t=[{cd_t0:.4f}, {cd_t1:.4f}])  |  FLATiron: {sim_cd_mean:.5f}  |  Ferziger Coarse: {g2_mean:.5f}  |  Ferziger Fine: {g4_mean:.5f}')


if __name__ == '__main__':
    plot_drag_lift()
