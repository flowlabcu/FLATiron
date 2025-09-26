import pytest
from flatiron_tk.physics import *
from flatiron_tk.mesh import RectMesh, LineMesh, CuboidMesh
from flatiron_tk.solver import NonLinearSolver
import numpy as np


def test_dofs(vector_equal):

    mesh = CuboidMesh(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1/4)

    st0 = SteadyScalarTransport(mesh, tag='c0')
    st0.set_element('CG', 1)
    st0.build_function_space()
    
    st1 = SteadyScalarTransport(mesh, tag='c1')
    st1.set_element('CG', 1)

    mphys = MultiphysicsProblem(st0, st1)
    mphys.set_element()
    mphys.build_function_space()

    V = mphys.get_function_space()
    V_st0, map_st0 = V.sub(0).collapse()
    V_st1, map_st1 = V.sub(1).collapse()

    local_st0 = V_st0.dofmap.index_map.size_local * V_st0.dofmap.index_map_bs
    map_st0 = np.asarray(map_st0)[:local_st0]
    dofs_st0_global_stripped = V.dofmap.index_map.local_to_global(map_st0)

    local_st1 = V_st1.dofmap.index_map.size_local * V_st1.dofmap.index_map_bs
    map_st1 = np.asarray(map_st1)[:local_st1]
    dofs_st1_global_stripped = V.dofmap.index_map.local_to_global(map_st1)

    dofs_st0_from_mphys_class = mphys.get_global_dofs(mphys.get_function_space(), 'c0')
    dofs_st1_from_mphys_class = mphys.get_global_dofs(mphys.get_function_space(), 'c1')

    assert vector_equal(dofs_st0_global_stripped, dofs_st0_from_mphys_class)
    assert vector_equal(dofs_st1_global_stripped, dofs_st1_from_mphys_class)
