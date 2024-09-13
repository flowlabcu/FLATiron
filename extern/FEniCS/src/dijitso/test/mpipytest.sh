#!/bin/sh

# Copyright (C) 2015 Martin Alnæs
# Copyright (C) 2015 Johannes Ring
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

# This script redirects output from a command to output.MYRANK

: ${PYTHON:=python3}

# See: http://docs.roguewave.com/threadspotter/2012.1/linux/manual_html/apas03.html
if [ ! -z $OMPI_COMM_WORLD_RANK ]; then
    export MYRANK=$OMPI_COMM_WORLD_RANK
elif [ ! -z $PMI_RANK ]; then
    export MYRANK=$PMI_RANK
else
    export MYRANK=`${PYTHON} -c "import dolfin;c=dolfin.mpi_comm_world();r=dolfin.MPI.rank(c);print(r);dolfin.MPI.barrier(c)"`
fi

CMD=${PYTHON}' -B -m pytest -svl --cov-report html --cov=dijitso --junitxml report$MYRANK.xml'
echo $CMD

OUT=output.$MYRANK
echo My rank is $MYRANK, writing to $OUT
if [ $MYRANK -eq 0 ]; then
  $CMD | tee $OUT
else
  $CMD > $OUT
fi

