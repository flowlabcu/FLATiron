"""
Reports the streamfunction value at the center of the cavity (x=0.5, y=0.5)
for each Rayleigh number, for comparison against Donea & Huerta (2003) Table 6.5.
"""

DONEA_PSI = [1.19, 5.15, 9.74, 17.32]

import numpy as np
from plot_verification import (
    RAYLEIGH_NUMBERS, _find_output_dir, _load_fields, _to_regular_grid,
    _streamfunction,
)

def main():

    computed_psi = []
    for Ra in RAYLEIGH_NUMBERS:
        output_dir = _find_output_dir(Ra)
        if output_dir is None:
            print(f"Ra={Ra:.0e}: no output directory found, skipping.")
            continue

        result = _load_fields(output_dir)
        if result is None:
            print(f"Ra={Ra:.0e}: no VTU files found, skipping.")
            continue

        coords_u, u_vals, coords_T, T_vals = result
        ux = u_vals[:, 0]

        xi, yi, Xi, Yi, Ux = _to_regular_grid(coords_u, ux)
        psi = _streamfunction(xi, yi, Ux)

        ix = np.argmin(np.abs(xi - 0.5))
        iy = np.argmin(np.abs(yi - 0.5))
        psi_center = np.abs(psi[iy, ix])
        psi_max = np.nanmax(np.abs(psi))

        computed_psi.append(psi_center)

    error = np.abs((np.array(computed_psi) - np.array(DONEA_PSI)) / np.array(DONEA_PSI)) * 100 
    print("Rayleigh number | Computed psi | Donea & Huerta psi | % Error")
    for Ra, psi_c, psi_d, err in zip(RAYLEIGH_NUMBERS, computed_psi, DONEA_PSI, error):
        print(f"{Ra:.0e} | {psi_c:.4f} | {psi_d:.4f} | {err:.4e}")



        
if __name__ == '__main__':
    main()
