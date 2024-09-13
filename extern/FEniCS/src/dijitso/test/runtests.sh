#!/usr/bin/env bash
set -e exit

: ${PYTHON:=python}

echo Running tests in serial:
export MYRANK=
export MPISIZE=
${PYTHON} -B -m pytest -svl --cov-report html --cov=dijitso --junitxml report$MYRANK.xml

for p in 1 4 8 16; do
  echo Running tests with mpi, n=$p
  export MPISIZE=$p
  mpirun -n $p ./mpipytest.sh
done
