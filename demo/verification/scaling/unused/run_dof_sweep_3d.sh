#!/bin/bash
# DOF/rank granularity sweep, fixed rank count -- runs at rank=1 (baseline)
# and rank=HIGH_RANK, at each of TARGET_DOFS_PER_RANK (defined inside
# flatiron_dof_sweep_3d.py / dolfinx_dof_sweep_3d.py), then plots efficiency
# vs. granularity. Same 3D unit-cube / tetrahedron problem and reference-
# matched KSP/AMG settings as run_scaling_3d_median.sh.
#
# Edit HIGH_RANK below to match physical core count on this machine (do not
# exceed physical cores -- see run_scaling_3d_median.sh for why).
set -e

source /home/njrovito/anaconda3/etc/profile.d/conda.sh
conda activate FLATironX

cd "$(dirname "$0")"

HIGH_RANK=4

rm -f flatiron_dof_sweep_3d.csv dolfinx_dof_sweep_3d.csv

echo "Baseline, 1 rank:"
mpirun --bind-to core -n 1 python3 flatiron_dof_sweep_3d.py
mpirun --bind-to core -n 1 python3 dolfinx_dof_sweep_3d.py

echo "$HIGH_RANK ranks:"
mpirun --bind-to core -n $HIGH_RANK python3 flatiron_dof_sweep_3d.py
mpirun --bind-to core -n $HIGH_RANK python3 dolfinx_dof_sweep_3d.py

python3 plot_dof_efficiency_3d.py --ranks $HIGH_RANK
