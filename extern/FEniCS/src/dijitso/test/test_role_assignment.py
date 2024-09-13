# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 Martin Sandve Alnæs
#
# This file is part of DIJITSO.
#
# DIJITSO is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DIJITSO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DIJITSO. If not, see <http://www.gnu.org/licenses/>.

import pytest
from dijitso.system import make_dirs
from dijitso.mpi import create_comms_and_role


@pytest.fixture()
def lib_dir0(comm):
    # Fake some common and some shared libdirs
    path = ".test_roles_%d" % (comm.rank,)
    make_dirs(path)
    return path


@pytest.fixture()
def lib_dir2(comm):
    # Fake some common and some shared libdirs
    path = ".test_roles_%d_of_2" % (comm.rank % 2,)
    make_dirs(path)
    return path


def test_role_root(comm, lib_dir2):
    buildon = "root"

    copy_comm, wait_comm, role = create_comms_and_role(comm, lib_dir2, buildon)

    if comm.rank == 0:
        expected_role = "builder"
    elif wait_comm.rank == 0:
        expected_role = "receiver"
    else:
        expected_role = "waiter"

    assert role == expected_role

    assert copy_comm is not None
    assert wait_comm is not None

    if role != "waiter":
        assert copy_comm.size == min(comm.size, 2)
    assert (comm.size // 2) <= wait_comm.size <= (comm.size // 2 + 1)


def test_role_node(comm, lib_dir2):
    buildon = "node"

    copy_comm, wait_comm, role = create_comms_and_role(comm, lib_dir2, buildon)

    if comm.rank in (0, 1):
        expected_role = "builder"
    else:
        expected_role = "waiter"

    assert role == expected_role

    assert copy_comm is None
    assert wait_comm is not None

    assert (comm.size // 2) <= wait_comm.size <= (comm.size // 2 + 1)


def test_role_process(comm, lib_dir0):
    buildon = "process"

    copy_comm, wait_comm, role = create_comms_and_role(comm, lib_dir0, buildon)

    expected_role = "builder"

    assert role == expected_role

    assert copy_comm is None
    assert wait_comm is None
