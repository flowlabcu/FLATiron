// Copyright (C) 2017 Benjamin Kehlet
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
// Unit tests for intersection construction

#include <dolfin/geometry/IntersectionConstruction.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/geometry/predicates.h>
#include <catch.hpp>

using namespace dolfin;


//-----------------------------------------------------------------------------
TEST_CASE("Intersection construction test")
{
  SECTION("failing case triangle_segment_3d")
  {
    const Point p0(0.333333333333333315, 0, 0.333333333333333315);
    const Point p1(0.66666666666666663, 0, 0.333333333333333315);
    const Point p2(0.66666666666666663, 0.333333333333333315, 0.333333333333333315);

    const Point q0(0.5, 0.25, 0.25);
    const Point q1(0.75, 0.25, 0.5);

    const Point ref_res(0.583333333333333259, 0.25, 0.333333333333333315);

    std::vector<Point> res = dolfin::IntersectionConstruction::intersection_triangle_segment_3d(p0, p1, p2,
												q0, q1);

    CHECK( res.size() == 1 );
    CHECK( (res[0] - ref_res).norm() < 1e-15 );
  }

  // SECTION("failing case tetrahedron_tetrahedron_3d")
  // {
  //   const Point p0(0.333333333333333315, 0, 0.333333333333333315);
  //   const Point p1(0.333333333333333315, 0, 0.66666666666666663);
  //   const Point p2(0.333333333333333315, 0.333333333333333315, 0.66666666666666663);
  //   const Point p3(0.66666666666666663, 0.333333333333333315, 0.66666666666666663);
  //   const Point q0(0.25, 0.25, 0.5);
  //   const Point q1(0.25, 0.25, 0.75);
  //   const Point q2(0.5, 0.25, 0.75);
  //   const Point q3(0.5, 0.5, 0.75);

  //   const std::vector<Point> res =
  //     IntersectionConstruction::intersection_tetrahedron_tetrahedron_3d(p0, p1, p2, p3,
  // 								      q0, q1, q2, q3);
  //   CHECK(res.size() == 5);
  // }
}
//-----------------------------------------------------------------------------
