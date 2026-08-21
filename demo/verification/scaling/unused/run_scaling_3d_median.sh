#!/bin/bash
# 3D (unit cube / tetrahedron, reference-matched KSP/AMG settings) scaling
# sweep with 5 trials per rank count, so plot_reference_comparison_3d.py can
# take the median instead of trusting a single noisy sample -- single-sample
# local timings showed up to ~40% run-to-run variance in earlier testing.
#
# Portable across machines: edit RANKS below to match physical core count
# (do not exceed physical cores -- SMT/hyperthreaded logical cores share
# execution units and will make scaling look artificially worse, which is a
# measurement artifact, not signal). --bind-to core pins each rank to a
# distinct physical core and avoids the scheduler spreading ranks across
# hyperthread siblings.
set -e

source /home/njrovito/anaconda3/etc/profile.d/conda.sh
conda activate FLATironX

cd "$(dirname "$0")"

RANKS="1 2 4 8"
TRIALS=5

rm -f flatiron_strong_scaling_3d.csv flatiron_weak_scaling_3d.csv
rm -f dolfinx_strong_scaling_3d.csv dolfinx_weak_scaling_3d.csv

for trial in $(seq 1 $TRIALS)
do
	for c in $RANKS
	do
		echo "Trial $trial, $c ranks:"
		mpirun --bind-to core -n $c python3 flatiron_scaling_3d.py
		mpirun --bind-to core -n $c python3 dolfinx_scaling_3d.py
	done
done

python3 plot_reference_comparison_3d.py
