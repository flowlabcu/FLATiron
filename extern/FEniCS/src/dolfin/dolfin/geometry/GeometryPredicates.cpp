// Copyright (C) 2016-2017 Anders Logg, August Johansson and Benjamin Kehlet
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
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
// First added:  2016-11-21
// Last changed: 2017-12-12

#include <cmath>
#include "CGALExactArithmetic.h"
#include "predicates.h"
#include "GeometryPredicates.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
bool GeometryPredicates::is_degenerate(const std::vector<Point>& simplex,
				       std::size_t tdim,
                                       std::size_t gdim)
{
  if (simplex.size() == tdim + 1)
  {
    if (tdim == 1)
    {
      return _is_degenerate_tdim_1(simplex);
    }
    else if (tdim == 2 and gdim == 2)
    {
      return _is_degenerate_tdim_2_gdim_2(simplex);
    }
    else if (tdim == 2 and gdim == 3)
    {
      return _is_degenerate_tdim_2_gdim_3(simplex);
    }
    else if (tdim == 3 and gdim == 3)
    {
      return _is_degenerate_tdim_3_gdim_3(simplex);
    }
  }
  else
  {
    dolfin_error("GeometryPredicates.cpp",
		 "evaluate is_degenerate",
		 "Input simplex size %d is not consistent with tdim %d", simplex.size(), tdim);
  }

  dolfin_error("GeometryPredicates.cpp",
	       "call is_degenerate",
	       "Not implemented for tdim %d, gdim %d", tdim, gdim);

  return false;
}
//-----------------------------------------------------------------------------
bool
GeometryPredicates::_is_degenerate_tdim_1
(const std::vector<Point>& simplex)
{
  return simplex[0] == simplex[1];
}
//-----------------------------------------------------------------------------
bool
GeometryPredicates::_is_degenerate_tdim_2_gdim_2
(const std::vector<Point>& simplex)
{
  return CHECK_CGAL(orient2d(simplex[0], simplex[1], simplex[2]) == 0.0,
		    is_degenerate_2d(simplex[0], simplex[1], simplex[2]));
}
//-----------------------------------------------------------------------------
bool
GeometryPredicates::_is_degenerate_tdim_2_gdim_3
(const std::vector<Point>& simplex)
{
  bool degen = true;
  const double ayz[2] = {simplex[0].y(), simplex[0].z()};
  const double byz[2] = {simplex[1].y(), simplex[1].z()};
  const double cyz[2] = {simplex[2].y(), simplex[2].z()};
  if (_orient2d(ayz, byz, cyz) != 0.0)
    degen = false;
  else
  {
    const double azx[2] = {simplex[0].z(), simplex[0].x()};
    const double bzx[2] = {simplex[1].z(), simplex[1].x()};
    const double czx[2] = {simplex[2].z(), simplex[2].x()};
    if (_orient2d(azx, bzx, czx) != 0.0)
      degen = false;
    else
    {
      const double axy[2] = {simplex[0].x(), simplex[0].y()};
      const double bxy[2] = {simplex[1].x(), simplex[1].y()};
      const double cxy[2] = {simplex[2].x(), simplex[2].y()};
      if (_orient2d(axy, bxy, cxy) != 0.0)
	degen = false;
    }
  }
  return CHECK_CGAL(degen,
		    is_degenerate_3d(simplex[0], simplex[1], simplex[2]));
}
//-----------------------------------------------------------------------------
bool
GeometryPredicates::_is_degenerate_tdim_3_gdim_3
(const std::vector<Point>& simplex)
{
  return
    CHECK_CGAL(orient3d(simplex[0], simplex[1], simplex[2], simplex[3]) == 0.0,
	       is_degenerate_3d(simplex[0], simplex[1], simplex[2], simplex[3]));
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::is_finite(const std::vector<Point>& simplex)
{
  for (auto p : simplex)
  {
    if (!std::isfinite(p.x()))
      return false;
    if (!std::isfinite(p.y()))
      return false;
    if (!std::isfinite(p.z()))
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::is_finite(const std::vector<double>& simplex)
{
  for (double p : simplex)
  {
    if (!std::isfinite(p))
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::convex_hull_is_degenerate(const std::vector<Point>& points,
                                                   std::size_t gdim)
{
  // Points are assumed to be unique

  if (points.size() < gdim+1)
    return true;

  if (gdim == 2)
  {
    // FIXME!
    return false;
  }
  else if (gdim == 3)
  {
    std::size_t i = 0, j = 1, k = 2;
    bool found = false;

    // Find three point which are not collinear
    for (; i < points.size(); i++)
    {
      for (j = i+1; j < points.size(); j++)
      {
        for (k = j+1; k < points.size(); k++)
        {
          const Point ij = points[j] - points[i];
          const Point ik = points[k] - points[i];
          if ( -(std::abs( (ij/ij.norm() ).dot(ik/ik.norm()))-1)  > DOLFIN_EPS)
          {
            found = true;
            break;
          }
        }
        if (found) break;
      }
      if (found)
        break;
    }

    // All points are collinear
    if (!found)
      return false;

    for (std::size_t l = 0; l < points.size();  l++)
    {
      if (l == i || l == j || l == k)
        continue;

      if (orient3d(points[i], points[j], points[k], points[l]) == 0.0)
        return true;
    }

    return false;
  }

  dolfin_error("GeometryPredicates.h",
               "call convex_hull_is_degenerate",
               "Only fully implemented for gdim == 3, not gdim = %d", gdim);
  return false;
}
//-----------------------------------------------------------------------------
