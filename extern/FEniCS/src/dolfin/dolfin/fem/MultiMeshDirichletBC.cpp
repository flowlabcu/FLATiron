// Copyright (C) 2014 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 4 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2014-05-12
// Last changed: 2016-03-02

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Array.h>
#include <dolfin/function/MultiMeshFunctionSpace.h>
#include <dolfin/function/Constant.h>
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include "DirichletBC.h"
#include "MultiMeshDirichletBC.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshDirichletBC(std::shared_ptr<const MultiMeshFunctionSpace> V,
                                           std::shared_ptr<const GenericFunction> g,
                                           std::shared_ptr<const SubDomain> sub_domain,
                                           std::string method,
                                           bool check_midpoint,
                                           bool exclude_overlapped_boundaries)
  : _function_space(V),
    _sub_domain(0),
    _exclude_overlapped_boundaries(exclude_overlapped_boundaries)
{
  log(PROGRESS, "Initializing multimesh Dirichlet boundary conditions.");

  // Initialize subdomain wrapper
  _sub_domain.reset(new MultiMeshSubDomain(sub_domain,
                                           V->multimesh(),
                                           _exclude_overlapped_boundaries));

  // Iterate over parts
  for (std::size_t part = 0; part < V->num_parts(); part++)
  {
    // Get view of function space for part
    std::shared_ptr<const FunctionSpace> V_part = V->view(part);

    // Create Dirichlet boundary condition for part
    std::shared_ptr<DirichletBC> bc(new DirichletBC(V_part,
                                                    g,
                                                    _sub_domain,
                                                    method,
                                                    check_midpoint));

    // HACK: function space dimension does not match tensor dimensions
    bc->parameters["check_dofmap_range"] = false;

    // Add to map
    _bcs[part] = bc;
  }
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshDirichletBC(std::shared_ptr<const MultiMeshFunctionSpace> V,
                                           std::shared_ptr<const GenericFunction> g,
                                           std::shared_ptr<const SubDomain> sub_domain,
                                           std::size_t part,
                                           std::string method,
                                           bool check_midpoint,
                                           bool exclude_overlapped_boundaries)
  : _function_space(V),
    _sub_domain(0),
    _exclude_overlapped_boundaries(exclude_overlapped_boundaries)
{
  log(PROGRESS, "Initializing multimesh Dirichlet boundary conditions.");

  // Initialize subdomain wrapper
  _sub_domain.reset(new MultiMeshSubDomain(sub_domain,
                                           V->multimesh(),
                                           _exclude_overlapped_boundaries));

  // Get view of function space for part
  std::shared_ptr<const FunctionSpace> V_part = V->view(part);

  // Create Dirichlet boundary condition for part
  std::shared_ptr<DirichletBC> bc(new DirichletBC(V_part,
                                                  g,
                                                  _sub_domain,
                                                  method,
                                                  check_midpoint));

  // HACK: function space dimension does not match tensor dimensions
  bc->parameters["check_dofmap_range"] = false;

  // Add to map
  _bcs[part] = bc;
}
//-----------------------------------------------------------------------------
class EmptySubDomain : public SubDomain {
  public:
    bool inside(Eigen::Ref<const Eigen::VectorXd> x, bool on_boundary) const {
      return false;
    }
};

MultiMeshDirichletBC::MultiMeshDirichletBC(std::shared_ptr<const MultiMeshFunctionSpace> V,
                                          std::shared_ptr<const GenericFunction> g,
                                          std::shared_ptr<const MeshFunction<std::size_t>> sub_domains,
                                          std::size_t sub_domain,
                                          std::size_t part,
                                          std::string method)
  : _function_space(V),
    _sub_domain(0),
    _exclude_overlapped_boundaries(false)
{
  log(PROGRESS, "Initializing multimesh Dirichlet boundary conditions.");

  // Initialize subdomain wrapper
  auto empty_subdomain = std::make_shared<EmptySubDomain>();
  _sub_domain.reset(new MultiMeshSubDomain(empty_subdomain,
                                           V->multimesh(),
                                           _exclude_overlapped_boundaries));


  // Construct BC for specified part
  std::shared_ptr<const FunctionSpace> V_part = V->view(part);
  std::shared_ptr<DirichletBC> bc(new DirichletBC(V_part,
                                                  g,
                                                  sub_domains,
                                                  sub_domain,
                                                  method));

  // HACK: function space dimension does not match tensor dimensions
  bc->parameters["check_dofmap_range"] = false;


  // Add to list
  _bcs[part] = bc;
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshDirichletBC(const MultiMeshDirichletBC& bc)
{
  *this = bc;

  // Iterate over boundary conditions and call the copy constructor
  for (const auto &pair : _bcs)
    _bcs[pair.first] = std::make_shared<DirichletBC>(*pair.second);
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::~MultiMeshDirichletBC()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericMatrix& A) const
{
  // Iterate over boundary conditions
  for (const auto &pair : _bcs)
  {
    // Set current part for subdomain wrapper
    dolfin_assert(_sub_domain);
    _sub_domain->set_current_part(pair.first);

    // Apply boundary condition for current part
    pair.second->apply(A);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericVector& b) const
{
  // Iterate over boundary conditions
  for (const auto &pair : _bcs)
  {
    // Set current part for subdomain wrapper
    dolfin_assert(_sub_domain);
    _sub_domain->set_current_part(pair.first);

    // Apply boundary condition
    pair.second->apply(b);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericMatrix& A,
                                 GenericVector& b) const
{
  // Iterate over boundary conditions
  for (const auto &pair : _bcs)
  {
    // Set current part for subdomain wrapper
    dolfin_assert(_sub_domain);
    _sub_domain->set_current_part(pair.first);

    // Apply boundary condition
    pair.second->apply(A, b);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericVector& b,
                                 const GenericVector& x) const
{
  // Iterate over boundary conditions
  for (const auto &pair : _bcs)
  {
    // Set current part for subdomain wrapper
    dolfin_assert(_sub_domain);
    _sub_domain->set_current_part(pair.first);

    // Apply boundary condition
    pair.second->apply(b, x);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericMatrix& A,
                                 GenericVector& b,
                                 const GenericVector& x) const
{
  // Iterate over boundary conditions
  for (const auto &pair : _bcs)
  {
    // Set current part for subdomain wrapper
    dolfin_assert(_sub_domain);
    _sub_domain->set_current_part(pair.first);

    // Apply boundary condition
    pair.second->apply(A, b, x);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::zero(GenericMatrix& A) const
{
  // Iterate over boundary conditions
  for (const auto &pair : _bcs)
  {
    // Set current part for subdomain wrapper
    dolfin_assert(_sub_domain);
    _sub_domain->set_current_part(pair.first);

    // Apply boundary condition for current part
    pair.second->zero(A);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::homogenize()
{
  // Iterate over boundary conditions
  for (const auto &pair : _bcs)
  {
    // Homogenize boundary condition
    pair.second->homogenize();
  }
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshSubDomain::MultiMeshSubDomain
(std::shared_ptr<const SubDomain> sub_domain,
 std::shared_ptr<const MultiMesh> multimesh,
 bool exclude_overlapped_boundaries)
  : _user_sub_domain(sub_domain),
    _multimesh(multimesh),
    _current_part(0),
    _exclude_overlapped_boundaries(exclude_overlapped_boundaries)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshSubDomain::~MultiMeshSubDomain()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool MultiMeshDirichletBC::MultiMeshSubDomain::inside(const Array<double>& x,
                                                      bool on_boundary) const
{
  dolfin_assert(_user_sub_domain);

  // If point is on boundary, check that it really is on the boundary,
  // which it may not be if it is contained in some other mesh.
  if (on_boundary && _exclude_overlapped_boundaries)
  {
    for (std::size_t part = 0; part < _multimesh->num_parts(); part++)
    {
      // Skip current part
      if (part == _current_part)
        continue;

      // Check whether point is contained in other mesh
      const Point point(x.size(), x.data());
      if (_multimesh->bounding_box_tree(part)->collides_entity(point))
      {
        on_boundary = false;
        break;
      }
    }
  }

  // Call user-defined function with possibly modified on_boundary
  return _user_sub_domain->inside(x, on_boundary);
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::MultiMeshSubDomain::set_current_part
(std::size_t current_part)
{
  _current_part = current_part;
}
//-----------------------------------------------------------------------------
