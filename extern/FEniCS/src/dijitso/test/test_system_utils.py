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

import io
import os
import uuid
from dijitso.system import lockfree_move_file, make_dirs


def test_get_status_output():
    pass  # FIXME


def test_lockfree_move_file():
    # Running this loop in multiple processes with mpi is perhaps a
    # decent test?

    # Fixed directory independent of mpi rank
    tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".test_lockfree_move_file")
    make_dirs(tmpdir)

    # source is different for each process and loop iteration
    n = 3
    u = uuid.uuid4().int
    srcs = [os.path.join(tmpdir, "test_lockfree_move_file-%d.src-%d" % (u, i))
            for i in range(n)]

    # fixed destination
    dst = os.path.join(tmpdir, "test_lockfree_move_file.dst")
    # try_delete_file(dst)   # can't do this when testing with mpi!

    for src in srcs:
        with io.open(src, "w", encoding="utf-8") as f:
            f.write("dummy")
        assert os.path.exists(src)
        lockfree_move_file(src, dst)
        assert os.path.exists(dst)
        assert not os.path.exists(src)

    with io.open(dst, encoding="utf-8") as f:
        dummy = f.read()
    assert dummy == "dummy"
