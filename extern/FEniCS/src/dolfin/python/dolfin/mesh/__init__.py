import ufl
import dolfin.cpp as cpp
from . import svgtools

# Functions to extend cpp.mesh.Mesh with


def ufl_cell(self):
    return ufl.Cell(self.cell_name(),
                    geometric_dimension=self.geometry().dim())


def ufl_coordinate_element(self):
    """Return the finite element of the coordinate vector field of this
    domain.

    """
    cell = self.ufl_cell()
    degree = self.geometry().degree()
    return ufl.VectorElement("Lagrange", cell, degree,
                             dim=cell.geometric_dimension())


def ufl_domain(self):
    """Returns the ufl domain corresponding to the mesh."""
    # Cache object to avoid recreating it a lot
    if not hasattr(self, "_ufl_domain"):
        self._ufl_domain = ufl.Mesh(self.ufl_coordinate_element(),
                                    ufl_id=self.ufl_id(),
                                    cargo=self)
    return self._ufl_domain


def geometric_dimension(self):
    """Returns geometric dimension for ufl interface"""
    return self.geometry().dim()


def _repr_html_(self):
    return cpp.io.X3DOM.html(self)


def _repr_svg_(self):
    return svgtools.mesh2svg(self)


# Extend cpp.mesh.Mesh class
cpp.mesh.Mesh.ufl_cell = ufl_cell
cpp.mesh.Mesh.ufl_coordinate_element = ufl_coordinate_element
cpp.mesh.Mesh.ufl_domain = ufl_domain
cpp.mesh.Mesh.geometric_dimension = geometric_dimension

cpp.mesh.Mesh._repr_html_ = _repr_html_
cpp.mesh.Mesh._repr_svg_ = _repr_svg_

# Clean-up
del ufl_cell, ufl_coordinate_element, ufl_domain, geometric_dimension, _repr_html_, _repr_svg_


# Extend cpp.mesh.MultiMesh class
def MultiMesh_ufl_domain(self):
    """Returns the ufl domain corresponding to the mesh."""
    # Cache object to avoid recreating it a lot
    if not hasattr(self, "_ufl_domain"):
        self._ufl_domain = ufl.Mesh(self.ufl_coordinate_element(), ufl_id=self.ufl_id(), cargo=self)
    return self._ufl_domain


def MultiMesh_mpi_comm(self):
    return self.part(0).mpi_comm()


def MultiMesh_type(self):
    return self.part(0).type()


def MultiMesh_ufl_cell(self):
    """Returns the ufl cell of the mesh."""
    gdim = self.part(0).geometry().dim()
    cellname = self.type().description(False)
    return ufl.Cell(cellname, geometric_dimension=gdim)


def MultiMesh_ufl_coordinate_element(self):
    "Return the finite element of the coordinate vector field of this domain."
    cell = self.ufl_cell()
    degree = self.part(0).geometry().degree()
    return ufl.VectorElement("Lagrange", cell, degree, dim=cell.geometric_dimension())


def MultiMesh_ufl_id(self):
    "Returns an id that UFL can use to decide if two objects are the same."
    return self.id()


cpp.mesh.MultiMesh.ufl_domain = MultiMesh_ufl_domain
cpp.mesh.MultiMesh.mpi_comm = MultiMesh_mpi_comm
cpp.mesh.MultiMesh.type = MultiMesh_type
cpp.mesh.MultiMesh.ufl_cell = MultiMesh_ufl_cell
cpp.mesh.MultiMesh.ufl_coordinate_element = MultiMesh_ufl_coordinate_element
cpp.mesh.MultiMesh.ufl_id = MultiMesh_ufl_id

del MultiMesh_mpi_comm, MultiMesh_type, MultiMesh_ufl_cell, MultiMesh_ufl_coordinate_element
