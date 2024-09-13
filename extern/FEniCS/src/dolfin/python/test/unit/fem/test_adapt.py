"""Unit tests for assembly"""

# Copyright (C) 2018 Jorgen S. Dokken
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
#


import pytest
import numpy
from dolfin import *

from dolfin_utils.test import fixture, skip_in_parallel


@skip_in_parallel
def test_mesh_adaptation():
    mesh = UnitSquareMesh(10,10)
    rmesh = adapt(mesh)
    assert(numpy.isclose(mesh.hmax()/rmesh.hmax(), 2))


@skip_in_parallel
def test_cell_function_adaptation():
    mesh = UnitSquareMesh(10,10)
    rmesh = adapt(mesh)
    mf = MeshFunction("size_t", mesh, mesh.topology().dim())
    rmesh = adapt(mesh)
    rmf = adapt(mf,rmesh)
    assert(rmf.size() / mf.size() == 4)


@skip_in_parallel
def test_facet_function_adaptation():
    # Facet Function refinement needs global parameter
    parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    mesh = UnitSquareMesh(10,10)
    rmesh = adapt(mesh)
    mf = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    rmf = adapt(mf, rmesh)
    assert(len(rmf.array())/len(mf.array())==3.875)
