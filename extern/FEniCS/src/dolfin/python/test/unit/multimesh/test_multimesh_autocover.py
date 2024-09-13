# -*- coding: utf-8 -*-
"""Unit tests for the MultiMeshFunction autocover functionality"""

# Copyright (C) 2018 Jørgen S. Dokken
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
#
# First added:  2018-04-03
# Last changed: 2018-04-03


from dolfin import *
from dolfin_utils.test import fixture, skip_in_parallel
import numpy


@fixture
def multimesh():
    mesh_0 = UnitSquareMesh(20, 20)
    mesh_1 = RectangleMesh(Point(0.23, 0.23),
                           Point(0.66, 0.66), 10, 10)
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    multimesh.build()
    return multimesh


@fixture
def a():
    return 0.3


@fixture
def b():
    return 0.6


@fixture
def complex_multimesh(a, b):
    meshes = [UnitSquareMesh(50, 50),
              RectangleMesh(Point(0.105, 0.1), Point(a, 0.905), 15, 15),
              RectangleMesh(Point(0.101, b), Point(0.903, 0.902), 15, 15),
              RectangleMesh(Point(b, 0.08), Point(0.91, 0.91), 15, 15),
              RectangleMesh(Point(0.102, 0.09), Point(0.92, a), 15, 13)]
    multimesh = MultiMesh()
    for mesh in meshes:
        multimesh.add(mesh)
    multimesh.build()
    return multimesh


@fixture
def V(multimesh):
    return MultiMeshFunctionSpace(multimesh, "CG", 1)


@fixture
def one(V):
    return interpolate(Constant(1), V)


@fixture
def complex_mesh_one(complex_multimesh):
    V = MultiMeshFunctionSpace(complex_multimesh, "CG", 1)
    return interpolate(Constant(1), V)


@skip_in_parallel
def test_auto_cover_all(multimesh, one):
    # Cover all cells
    multimesh.auto_cover(0, Point(0, 0))
    multimesh.auto_cover(1, Point(0.5, 0.5))
    assert(numpy.isclose(assemble_multimesh(one * dX), 0))

    # Reset auto_cover
    multimesh.build()
    assert(numpy.isclose(assemble_multimesh(one * dX), 1))


@skip_in_parallel
def test_auto_cover_top(multimesh, one):
    # Cover cells on top mesh
    multimesh.auto_cover(0, Point(0, 0))
    assert(numpy.isclose(assemble_multimesh(one * dX), 0.1849))


@skip_in_parallel
def test_auto_cover_complex(a, b, complex_multimesh, complex_mesh_one):
    # Cover cells inscribed by the four top meshes
    complex_multimesh.auto_cover(0, Point(a + 0.5 * (b - a),
                                          a + 0.5 * (b - a)))
    assert(numpy.isclose(assemble_multimesh(complex_mesh_one * dX),
                         1 - (b - a)**2))
