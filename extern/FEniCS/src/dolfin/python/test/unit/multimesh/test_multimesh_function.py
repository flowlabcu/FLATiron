"""Unit tests for MultiMesh Functions"""

# Copyright (C) 2017 Simon Funke
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
# First added:  2017-12-12
# Last changed: 2017-12-12

import pytest
from dolfin import *

from dolfin_utils.test import skip_in_parallel

@skip_in_parallel
def test_multimesh_function():

    # Define meshes
    mesh_0 = UnitSquareMesh(2, 2)
    mesh_1 = RectangleMesh(Point(0.1*DOLFIN_PI, 0.1*DOLFIN_PI),
                           Point(0.2*DOLFIN_PI, 0.2*DOLFIN_PI),
                           2, 2)

    # Build multimesh
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    multimesh.build()

    V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)
    W = MultiMeshFunctionSpace(multimesh, "Lagrange", 2)

    # Create some basic multimesh functions
    FacetNormal(multimesh)

    TrialFunction(V)
    TestFunction(V)
    MultiMeshFunction(V)

    TrialFunction(W)
    TestFunction(W)
    MultiMeshFunction(W)
