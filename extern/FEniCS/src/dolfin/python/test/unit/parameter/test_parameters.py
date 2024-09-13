"""Unit tests for parameter library"""

# Copyright (C) 2011 Anders Logg
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

import pytest
import os

from dolfin import *
from dolfin_utils.test import *

xfail = pytest.mark.xfail(strict=True)

@skip_in_parallel
def test_simple(tempdir):

    # Create some parameters
    p0 = Parameters("test")
    p0.add("filename", "foo.txt")
    p0.add("maxiter", 100)
    p0.add("tolerance", 0.001)
    p0.add("monitor_convergence", True)

    # Save to file
    f0 = File(os.path.join(tempdir, "test_parameters.xml"))
    f0 << p0

    # Read from file
    p1 = Parameters()
    f1 = File(os.path.join(tempdir, "test_parameters.xml"))
    f1 >> p1

    # Check values
    assert p1.name() == "test"
    assert p1["filename"] == "foo.txt"
    assert p1["maxiter"] == 100
    assert p1["tolerance"] == 0.001
    assert p1["monitor_convergence"] == True

@skip_in_parallel
def test_gzipped_simple(tempdir):

    # Create some parameters
    p0 = Parameters("test")
    p0.add("filename", "foo.txt")
    p0.add("maxiter", 100)
    p0.add("tolerance", 0.001)
    p0.add("monitor_convergence", True)

    # Save to file
    f0 = File(os.path.join(tempdir, "test_parameters.xml.gz"))
    f0 << p0

    # Read from file
    p1 = Parameters()
    f1 = File(os.path.join(tempdir, "test_parameters.xml.gz"))
    f1 >> p1

    # Check values
    assert p1.name() == "test"
    assert p1["filename"] == "foo.txt"
    assert p1["maxiter"] == 100
    assert p1["tolerance"] == 0.001
    assert p1["monitor_convergence"] == True

@skip_in_parallel
def test_nested(tempdir):

    # Create some nested parameters
    p0 = Parameters("test")
    p00 = Parameters("sub0")
    p00.add("filename", "foo.txt")
    p00.add("maxiter", 100)
    p00.add("tolerance", 0.001)
    p00.add("monitor_convergence", True)
    p0.add("foo", "bar")
    p01 = Parameters(p00);
    p01.rename("sub1");
    p0.add(p00)
    p0.add(p01)

    # Save to file
    f0 = File(os.path.join(tempdir, "test_parameters.xml"))
    f0 << p0

    # Read from file
    p1 = Parameters()
    f1 = File(os.path.join(tempdir, "test_parameters.xml"))
    f1 >> p1

    # Check values
    assert p1.name() == "test"
    assert p1["foo"] == "bar"
    assert p1["sub0"]["filename"] == "foo.txt"
    assert p1["sub0"]["maxiter"] == 100
    assert p1["sub0"]["tolerance"] == 0.001
    assert p1["sub0"]["monitor_convergence"] == True

@skip_in_parallel
def test_parameters_update(tempdir):
    p0 = parameters.copy()
    p1 = parameters.copy()
    p1.update(p0)

@skip_in_parallel
def test_nested_read_existing(tempdir):
    """Test that we can read in a nested parameter database into
       an existing (and matching) parameter database"""

    file = File(os.path.join(tempdir, "test_parameters.xml"))
    file << parameters

    p = Parameters("test")
    file >> p
    file >> p

@skip_in_parallel
def test_solver_parameters():
    "Test that global parameters are propagated to solvers"

    # Record default values so we can change back
    absolute_tolerance = parameters["krylov_solver"]["absolute_tolerance"]

    # Set global parameters
    parameters["krylov_solver"]["absolute_tolerance"] = 1.23456

    # Create solvers
    krylov_solver = KrylovSolver()

    # Check that parameters propagate to solvers
    assert krylov_solver.parameters["absolute_tolerance"] == 1.23456

    # Reset parameters so that other tests will continue to work
    if absolute_tolerance is not None:
        parameters["krylov_solver"]["absolute_tolerance"] = absolute_tolerance

def test_parse_parameters():

    def check_pset_and_dict_args(pset, pset_dict):
        for k, v in pset_dict.items():
            assert pset[k] == v

    def create_command_line_args(pset_dict):
        # Construct args like: --a=1 --b=2 --c=3
        command_line_args1 = ["--" + str(k) + "=" + str(v) for k, v in pset_dict.items()]
        # Construct args like: --a 1 --b 2 --c 3
        command_line_args2 = [str(element) for k in pset_dict for element in (k, pset_dict[k])]
        return command_line_args1, command_line_args2

    pset_dict = {
        "integer": 1,
        "double": 2.5,
        "string": "string",
        "boolean": False
    }

    pset = Parameters("test_parse")
    for k, v in pset_dict.items():
        pset.add(k, v)

    command_line_args = create_command_line_args(pset_dict)
    pset.parse([""] + command_line_args[0])
    check_pset_and_dict_args(pset, pset_dict)
    pset.parse([""] + command_line_args[1])
    check_pset_and_dict_args(pset, pset_dict)

    changed_pset_dict = {
        "integer": 2,
        "double": 4.5,
        "string": "another string",
        "boolean": True
    }

    changed_command_line_args = create_command_line_args(changed_pset_dict)
    pset.parse([""] + changed_command_line_args[0])
    check_pset_and_dict_args(pset, changed_pset_dict)
    pset.parse([""] + changed_command_line_args[1])
    check_pset_and_dict_args(pset, changed_pset_dict)

@pytest.mark.parametrize('arg_factory', [pytest.param(True, marks=xfail),
                                         pytest.param(1, marks=xfail),
                                         pytest.param(1., marks=xfail),
                                         pytest.param("string", marks=xfail),
                                         pytest.param("none", marks=xfail),
                                         pytest.param(parameters, marks=xfail)
                                         ])
def test_wrong_parameters_setitem(arg_factory):
    value = arg_factory
    parameters["misplelled"] = arg_factory

@xfail
def test_wrong_parameters_getitem(arg_factory):
    parameters["misplelled"]
