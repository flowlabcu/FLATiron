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

#ifndef __GEOMETRY_PREDICATES_H
#define __GEOMETRY_PREDICATES_H

#include <vector>
#include <dolfin/log/LogStream.h>
#include "Point.h"

namespace dolfin
{

  /// This class implements geometric predicates, i.e. function that
  /// return either true or false.

  class GeometryPredicates
  {
  public:

    /// Check whether a single simplex is degenerate
    static bool is_degenerate(const std::vector<Point>& simplex,
			      std::size_t tdim,
			      std::size_t gdim);

    /// Check whether simplex is finite (not Inf or NaN)
    static bool is_finite(const std::vector<Point>& simplex);

    /// Check whether simplex is finite (not Inf or NaN)
    static bool is_finite(const std::vector<double>& simplex);

    /// Check whether the convex hull is degenerate
    static bool convex_hull_is_degenerate(const std::vector<Point>& p,
                                          std::size_t gdim);

  private:

    // Implementation of is_degenerate predicates
    static bool _is_degenerate_tdim_1(const std::vector<Point>& simplex);

    static bool _is_degenerate_tdim_2_gdim_2(const std::vector<Point>& simplex);

    static bool _is_degenerate_tdim_2_gdim_3(const std::vector<Point>& simplex);

    static bool _is_degenerate_tdim_3_gdim_3(const std::vector<Point>& simplex);
  };

}

#endif
