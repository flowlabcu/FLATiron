"""
Plots streamlines (streamfunction contours) and isothermal lines from
free-convection simulation output VTU files, for comparison against
Donea & Huerta (2003) Fig. 6.17 and Fig. 6.18.

Usage:
    python plot_verification.py

Expects output directories: output_Ra1e3/, output_Ra1e4/, etc.
Falls back to a single 'output/' directory if per-Ra dirs are absent.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.integrate import cumulative_trapezoid

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), 'output')
RAYLEIGH_NUMBERS = [1e3, 1e4, 1e5, 1e6]
RA_LABELS = [r'$Ra = 1e+03$', r'$Ra = 1e+04$', r'$Ra = 1e+05$', r'$Ra = 1e+06$']
GRID_N = 200  # interpolation resolution


def _ra_tag(Ra):
    exp = int(round(np.log10(Ra)))
    return f'Ra1e+{exp:02d}'


def _find_output_dir(Ra):
    """Return the output directory for a given Ra, or None if not found."""
    per_ra = os.path.join(os.path.dirname(__file__), f'output-{_ra_tag(Ra)}')
    if os.path.isdir(per_ra):
        return per_ra
    return None


def _last_step_index(output_dir, prefix):
    """Find the highest step index for files like <prefix>_p0_XXXXXX.vtu."""
    files = [f for f in os.listdir(output_dir)
             if f.startswith(f'{prefix}_p0_') and f.endswith('.vtu')]
    if not files:
        return None
    indices = [int(f.split('_p0_')[1].split('.vtu')[0]) for f in files]
    return max(indices)


def _parse_vtu(path, field_name):
    """Parse a VTU file and return (coords_xy, field_values), dropping ghost points."""
    tree = ET.parse(path)
    root = tree.getroot()
    piece = root.find('.//Piece')

    # Coordinates
    pts_data = piece.find('Points/DataArray').text.strip().split()
    pts = np.array(pts_data, dtype=float).reshape(-1, 3)
    coords = pts[:, :2]  # x, y

    # Field + ghost mask
    pd = piece.find('PointData')
    vals = None
    ghost = None
    for da in pd:
        name = da.attrib.get('Name')
        if name == field_name:
            ncomp = int(da.attrib.get('NumberOfComponents', '1'))
            vals = np.array(da.text.strip().split(), dtype=float)
            if ncomp > 1:
                vals = vals.reshape(-1, ncomp)
        elif name == 'vtkGhostType':
            ghost = np.array(da.text.strip().split(), dtype=int)

    if vals is None:
        raise ValueError(f"Field '{field_name}' not found in {path}")

    if ghost is not None:
        keep = ghost == 0
        coords = coords[keep]
        vals = vals[keep]

    return coords, vals


def _piece_paths(output_dir, prefix, step):
    """Return every rank's VTU piece file for a given field/step (u_p0_..., u_p1_..., ...)."""
    files = [f for f in os.listdir(output_dir)
             if f.startswith(f'{prefix}_p') and f.endswith(f'_{step:06d}.vtu')]
    files.sort(key=lambda f: int(f.split(f'{prefix}_p')[1].split('_')[0]))
    return [os.path.join(output_dir, f) for f in files]


def _load_all_pieces(output_dir, prefix, step, field_name):
    """Load and concatenate a field across all MPI-rank pieces for one timestep."""
    coords_list, vals_list = [], []
    for path in _piece_paths(output_dir, prefix, step):
        c, v = _parse_vtu(path, field_name)
        coords_list.append(c)
        vals_list.append(v)
    return np.concatenate(coords_list, axis=0), np.concatenate(vals_list, axis=0)


def _load_fields(output_dir):
    """Load u (velocity) and T (temperature) from last timestep in output_dir,
    concatenating all MPI-rank pieces."""
    step_u = _last_step_index(output_dir, 'u')
    step_T = _last_step_index(output_dir, 'T')
    if step_u is None or step_T is None:
        return None

    coords_u, u_vals = _load_all_pieces(output_dir, 'u', step_u, 'u')
    coords_T, T_vals = _load_all_pieces(output_dir, 'T', step_T, 'T')

    return coords_u, u_vals, coords_T, T_vals


def _to_regular_grid(coords, values, n=GRID_N):
    """Interpolate unstructured data to a regular n×n grid on [0,1]×[0,1]."""
    eps = 1e-3  # stay just inside the boundary to avoid extrapolation artifacts
    xi = np.linspace(eps, 1 - eps, n)
    yi = np.linspace(eps, 1 - eps, n)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(coords, values, (Xi, Yi), method='linear')
    return xi, yi, Xi, Yi, Zi


def _streamfunction(xi, yi, ux_grid):
    """
    Compute streamfunction ψ by integrating u_x along y:
        ψ(x, y) = ∫₀ʸ u_x(x, y') dy'
    ux_grid has shape (ny, nx) with rows = y-axis.
    """
    psi = cumulative_trapezoid(ux_grid, yi, axis=0, initial=0)
    return psi


def _plot_panel(ax, Xi, Yi, Z, levels, title, colors='k'):
    """Draw a contour panel with labeled contours, matching the book style."""
    cs = ax.contour(Xi, Yi, Z, levels=levels, colors=colors,
                    linewidths=0.8, linestyles='solid')
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.1f')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', which='both', labelsize=10, width=1.5)
    ax.set_xlabel('x', fontsize=10, fontweight='bold')
    ax.set_ylabel('y', fontsize=10, fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


def main():
    any_plotted = False

    for idx, Ra in enumerate(RAYLEIGH_NUMBERS):
        ra_tag = _ra_tag(Ra)
        output_dir = _find_output_dir(Ra)
        if output_dir is None:
            print(f"No output directory found for Ra={Ra:.0e}, skipping.")
            continue

        result = _load_fields(output_dir)
        if result is None:
            print(f"No VTU files found for Ra={Ra:.0e} in {output_dir}, skipping.")
            continue

        coords_u, u_vals, coords_T, T_vals = result
        print(f"Plotting Ra={Ra:.0e} from {output_dir}")

        ux = u_vals[:, 0]
        uy = u_vals[:, 1]

        xi, yi, Xi, Yi, Ux = _to_regular_grid(coords_u, ux)
        _, _, _, _, Uy = _to_regular_grid(coords_u, uy)
        _, _, _, _, T_grid = _to_regular_grid(coords_T, T_vals)

        psi = _streamfunction(xi, yi, Ux)

        # Streamfunction levels: match the book's contour values per Ra
        psi_levels = _streamfunction_levels(Ra, psi)
        T_levels = _temperature_levels(Ra)

        fig_stream, ax_stream = plt.subplots(figsize=(4, 4))
        _plot_panel(ax_stream, Xi, Yi, psi, psi_levels, RA_LABELS[idx])
        fig_stream.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), f'streamlines_{ra_tag}.png')
        fig_stream.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig_stream)
        print(f"Saved {out_path}")

        fig_temp, ax_temp = plt.subplots(figsize=(4, 4))
        _plot_panel(ax_temp, Xi, Yi, T_grid, T_levels, RA_LABELS[idx])
        fig_temp.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), f'isothermals_{ra_tag}.png')
        fig_temp.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig_temp)
        print(f"Saved {out_path}")

        any_plotted = True

    if not any_plotted:
        print("No data found. Run the simulation first.")


def _streamfunction_levels(Ra, psi):
    """
    Choose contour levels for the streamfunction to match the book's figures.
    For Ra=1e3: max ψ ≈ 1.1; book shows ±0.1 to ±1.1 steps of 0.1
    For Ra=1e4: max ψ ≈ 5;   book shows integer levels
    For Ra=1e5: max ψ ≈ 9;   book shows steps ~1-2
    For Ra=1e6: max ψ ≈ 17;  book shows steps ~2
    Use symmetric levels around zero derived from the actual solution max.
    """
    psi_max = np.nanmax(np.abs(psi))
    if psi_max < 1e-10:
        return np.array([0.0])

    if Ra <= 1e3:
        step = 0.1
    elif Ra <= 1e4:
        step = 0.5
    elif Ra <= 1e5:
        step = 1.0
    else:
        step = 2.0

    n_levels = max(int(psi_max / step), 2)
    pos = np.arange(step, n_levels * step + step * 0.5, step)
    levels = np.concatenate([-pos[::-1], [0], pos])
    return levels[np.abs(levels) <= psi_max * 1.05]


def _temperature_levels(Ra):
    """
    Isothermal contour levels.  Temperature range is [-0.5, 0.5].
    The book shows ~10 isotherms from -0.4 to 0.4 for all Ra.
    """
    if Ra <= 1e3:
        return np.arange(-0.4, 0.45, 0.1)
    else:
        return np.concatenate([
            np.arange(-0.4, -0.05, 0.1),
            np.arange(0.0, 0.45, 0.1)
        ])


if __name__ == '__main__':
    main()
