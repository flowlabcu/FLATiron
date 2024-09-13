#!/usr/bin/env bash
set -e exit

: ${PYTHON:=python}

while [ $? -eq 0 ]
do
  for p in 8; do
    echo RUNNING WITH p=$p
    mpirun -n $p xterm -e gdb -ex r -ex q -args ${PYTHON} -B -m pytest -svl
  done
done
