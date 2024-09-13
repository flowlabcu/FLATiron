# -*- coding: utf-8 -*-
"""This module handles the MultiMeshFunction class in Python.
"""
# Copyright (C) 2017 Jørgen Schartum Dokken
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

import ufl
import dolfin.cpp as cpp

from dolfin.function.multimeshfunctionspace import MultiMeshFunctionSpace
from dolfin.function.function import Function


class MultiMeshFunction(ufl.Coefficient):
    r"""This class represents a multimeshfunction
    :math:`u_h=(u_{h,1}\cross \dots u_{h,N}` in a finite
    element multimeshfunction space
    :math:`V_h=V_{h,1}\cross \dots V_{h,N}`, given by

    .. math::

        u_{h,j}=  \sum_{i=1}^n U_i \phi_{i,j},

    where :math:`\{\phi_{i,j}\}_{i=1}^n` is a basis for :math:`V_{h,j}`,
    and :math:`U` is a vector of expansion coefficients for
    :math:`u_h`.

    *Arguments*
        There is a maximum of two arguments. The first argument must be a
        :py:class:`MultiMeshFunctionSpace
         <dolfin.cpp.function.MultiMeshFunctionSpace>`.

        The second argument must be a GenericVector and is intended for library
        use only.

    *Examples*
        Create a MultiMeshFunction:

        - from a :py:class:`MultiMeshFunctionSpace
          <dolfin.cpp.function.MultiMeshFunctionSpace>` ``V``

          .. code-block:: python

              f = MultiMeshFunction(V)


        - from a :py:class:`MultiMeshFunctionSpace
          <dolfin.cpp.function.MultiMeshFunctionSpace>` ``V`` and a
          :py:class:`GenericVector <dolfin.cpp.GenericVector>` ``v``

          *Warning: this constructor is intended for internal libray use only.*

          .. code-block:: python

              g = MultiMeshFunction(V, v)

    """

    def __init__(self, *args, **kwargs):
        """Initialize MultiMeshFunction."""
        # Initial quick check for valid arguments (other checks
        # sprinkled below)
        if len(args) == 0:
            raise TypeError("expected 1 or more arguments")
        # Type switch on argument types
        if isinstance(args[0], MultiMeshFunction):
            other = args[0]
            if len(args) == 1:
                # Copy constructor used to be here
                raise RuntimeError("Use 'MultiMeshFunction.copy(deepcopy=True)' for copying.")
            else:
                raise NotImplementedError
        elif isinstance(args[0], cpp.function.MultiMeshFunction):
            raise NotImplementedError
        elif isinstance(args[0], MultiMeshFunctionSpace):
            V = args[0]
            # If initialising from a FunctionSpace
            if len(args) == 1:
                # If passing only the FunctionSpace
                self._cpp_object = cpp.function.MultiMeshFunction(V._cpp_object)
                ufl.Coefficient.__init__(self, V._parts[0].ufl_function_space(),
                                         count=self._cpp_object.id())
            elif len(args) == 2:
                other = args[1]
                if isinstance(other, cpp.function.MultiMeshFunction):
                    raise NotImplementedError
                else:
                    self._cpp_object = cpp.function.MultiMeshFunction.__init__(self, V, other)
                    ufl.Coefficient.__init__(self, V._parts[0]
                                             .ufl_function_space(),
                                             count=self._cpp_object.id())

            else:
                raise TypeError("too many arguments")

            # Keep a reference of the functionspace with additional attributes
            self._V = V
        else:
            raise TypeError("expected a MultiMeshFunctionSpace or a MultiMeshFunction as argument 1")

    def vector(self):
        return self._cpp_object.vector()

    def function_space(self):
        return self._V

    def num_parts(self):
        return self._V.num_parts()

    def assign(self, rhs):
        """
        Parameters:
            rhs: A dolfin.MultiMeshFunction
        """
        # Assign a MultiMeshFunction into a MultiMeshFunction
        if isinstance(rhs, MultiMeshFunction):
            self._cpp_object.vector()[:] = rhs.vector()[:]
        else:
            raise TypeError("expected a MultiMeshFunction as argument.")

    def part(self, i, deepcopy=False):
        f = Function(self._cpp_object.part(i, deepcopy))
        f.rename(self._cpp_object.name(), self._cpp_object.label())
        return f

    def parts(self, deepcopy=False):
        """
        Generator for MultiMeshFunction
        """
        for part in range(self._V._cpp_object.multimesh().num_parts()):
            yield self.part(part, deepcopy)

    def assign_part(self, part, function):
        self._cpp_object.assign_part(part, function._cpp_object)

    def interpolate(self, v):
        """
        Interpolate function.

        *Arguments*
            v
              a :py:class:`MultiMeshFunction <dolfin.functions.function.MultimeshFunction>` or
              an :py:class:`Expression <dolfin.functions.expression.Expression'>
        *Example of usage*
            .. code-block:: python

                V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)
                v = MultiMeshFunction(V)
                w = Expression("sin(pi*x[0])")
                v.interpolate(w)
        """
        # Developer note: This version involves a lot of copying
        # and should be changed at some point.
        # Developer note: Interpolate does not set inactive dofs to zero,
        # and should be fixed
        # Check argument
        if isinstance(v, MultiMeshFunction):
            # Same multimesh required for interpolation
            # Developer note: Is this test necessary?
            if self._V.multimesh().id() != v._V.multimesh().id():
                raise RuntimeError("MultiMeshFunctions must live on same MultiMesh")
            for i, vp in enumerate(self.parts(deepcopy=True)):
                vm = v.part(i, deepcopy=True)
                vp.interpolate(vm)
                self._cpp_object.assign_part(i, vp._cpp_object)

        elif (isinstance(v, (ufl.Coefficient, cpp.function.Expression))):
            for i, vp in enumerate(self.parts(deepcopy=True)):
                vp.interpolate(v)
                self._cpp_object.assign_part(i, vp._cpp_object)

        elif isinstance(v, cpp.function.MultiMeshFunction):
            for i, vp in enumerate(self.parts(deepcopy=True)):
                vm = v.part(i)
                vp.interpolate(vm)
                self._cpp_object.assign_part(i, vp._cpp_object)

        else:
            raise TypeError("Expected an Expression or a MultiMeshFunction.")
        # Set inactive dofs to zero
        for part in range(self._V.num_parts()):
            dofs = self._V.dofmap().inactive_dofs(self._V.multimesh(), part)
            self._cpp_object.vector()[dofs] = 0

    def sub(self, i, deepcopy=False):
        """
        Return a sub function
        The sub functions are numbered from i = 0..N-1, where N is the
        total number of sub spaces,

        *Arguments*
            i : int
                The number of the  sub function
        """
        if not isinstance(i, int):
            raise TypeError("expects an 'int' as first argument")
        num_sub_spaces = self.function_space().info[0].num_sub_elements()
        if num_sub_spaces == 1:
            raise RuntimeError("No subfunctions to extract")
        if not i < num_sub_spaces:
            raise RuntimeError("Can only extract subfunctions with i = 0..%d"
                               % num_sub_spaces)
        if deepcopy:
            sub_space = MultiMeshFunctionSpace(self.function_space()
                                               .multimesh(),
                                               self.function_space()
                                               .info[0].sub_elements()[i])
            mmf = MultiMeshFunction(sub_space)
            for j in range(self.num_parts()):
                mmf.assign_part(j, self.part(j).sub(i))
            return mmf
        else:
            raise NotImplementedError

    def split(self, deepcopy=False):
        """Extract any sub functions.

        A sub function can be extracted from a discrete function that
        is in a mixed, vector, or tensor MultiMeshFunctionSpace. The sub
        function resides in the subspace of the mixed space.

        *Arguments*
            deepcopy
                Copy sub function vector instead of sharing

        """

        num_sub_spaces = self.function_space().info[0].num_sub_elements()
        if num_sub_spaces == 1:
            raise RuntimeError("No subfunctions to extract")
        return tuple(self.sub(i, deepcopy) for i in range(num_sub_spaces))
