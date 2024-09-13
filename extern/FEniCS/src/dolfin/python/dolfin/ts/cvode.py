# -*- coding: utf-8 -*-
"""This module describes the Python wrapper of the cpp.CVode"""

# Copyright (C) 2013 Johan Hake
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
# First added:  2013-04-24
# Last changed: 2013-04-24

# Import SWIG-generated extension module (DOLFIN C++)
import dolfin.cpp as cpp

__all__ = ["CVode"]


class CVode(cpp.ts.CVode):
    def __init__(self, scheme):
        """
        Construct a CVode

        *Arguments*
            cv_lmm (LMM)
                FIX: add description
            cv_iter (ITER)
                FIX: add description
        """
        self._scheme = scheme
        cpp.ts.CVode.__init__(self, scheme)
