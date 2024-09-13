# Copyright (C) 2017 Jørgen S. Dokken
#
# Distributed under the terms of the GNU Lesser Public License (LGPL),
# either version 3 of the License, or (at your option) any later
# version.

import types
import ufl
import dolfin.cpp as cpp
from dolfin.function.constant import Constant
from dolfin.function.multimeshfunctionspace import MultiMeshFunctionSpace
from dolfin import MultiMeshSubSpace
from dolfin.fem.projection import project


class MultiMeshDirichletBC(cpp.fem.MultiMeshDirichletBC):
    # Arguments:
    # V : MultiMeshSubSpace or MultiMeshFunctionSpace
    # value: Expression/MultiMeshFunction/Constant
    #
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            if not isinstance(args[0], cpp.fem.MultiMeshDirichletBC):
                raise RuntimeError("Expecting a MultiMeshDirichletBC as only argument for copy constructor")
            cpp.fem.MultiMeshDirichletBC.__init__(self, args[0])
            return
        if not isinstance(args[0], (MultiMeshFunctionSpace, MultiMeshSubSpace)):
            raise RuntimeError("Expecting MultiMeshFunctionSpace or MultiMeshSubSpace as first argument")

        if len(args) >= 2:
            # Check if we have an UFL-expression or a concrete type
            if not hasattr(args[1], "_cpp_object"):
                if isinstance(args[1], ufl.classes.Expr):
                    expr = project(args[1], args[0])  # Should be interpolation
                else:
                    expr = Constant(args[1])
                args = args[:1] + (expr, 1) + args[2:]
        if isinstance(args[1], (float, int)):
            u = cpp.function.Constant(float(args[1]))
        elif isinstance(args[1], ufl.Coefficient):
            u = args[1]._cpp_object
        elif isinstance(args[1], cpp.function.GenericFunction):
            u = args[1]
        else:
            raise RuntimeError("Second argument must be convertiable to a GenericFucntion")

        if isinstance(args[0], MultiMeshSubSpace):
            args = args[:1] + (u,) + args[2:]

        else:
            args = args[:1] + (u,) + args[2:]
            args = (args[0]._cpp_object,) + args[1:]

        if len(args) >= 3 and isinstance(args[2], types.FunctionType):
            raise NotImplementedError("User-specified subdomains not implemented")
        if isinstance(args[2], cpp.mesh.SubDomain):
            self.sub_domain = args[2]
            args = args[:2] + (self.sub_domain,) + args[3:]
        elif isinstance(args[2], str):
            raise NotImplementedError("User-specified subdomains not implemented")
        elif isinstance(args[2], cpp.mesh.MeshFunctionSizet):
            pass
        else:
            raise RuntimeError("Invalid argument")

        # Add kwargs
        if isinstance(args[-1], str):
            method = args[-1]
        else:
            method = kwargs.pop("method", "topological")
            args += (method,)
        check_midpoint = kwargs.pop("check_midpoint", None)
        if check_midpoint is not None:
            args += (check_midpoint,)

        if (len(kwargs) > 0):
            raise RuntimeError("Invalid keyword arguments", kwargs)

        super().__init__(*args)
