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

from pkg_resources import get_distribution

"""This is dijitso -- a lightweight distributed just-in-time shared
library builder."""

__author__ = "Martin Sandve Alnæs"
__version__ = get_distribution('fenics-dijitso').version

__all__ = ["validate_params", "jit", "extract_factory_function",
           "set_log_level"]

from dijitso.params import validate_params
from dijitso.jit import jit, DijitsoError
from dijitso.jit import extract_factory_function
from dijitso.log import set_log_level

# Import main function, entry point to script
from dijitso.__main__ import main
