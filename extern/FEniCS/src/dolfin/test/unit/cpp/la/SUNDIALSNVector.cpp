// Copyright (C) 2017
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
// First added:  2017-03-08
// Last changed: 2017-03-08
//
// Unit tests Selected methods for SUNDIALSNVector

#include <dolfin.h>
#include <gtest/gtest.h>

using namespace dolfin;

//----------------------------------------------------
void _test(MPI_Comm comm)
{
  SUNDIALSNVector v(comm, 10), u(comm, 10);
  v.N_VConst(0,v.nvector());
  u.N_VConst(0.0,u.nvector());
  ASSERT_EQ(v.vec()->sum(), 0.0);

  // N_VConst(double a, N_Vector x)
  v.N_VConst(1.0,v.nvector());
  ASSERT_EQ(v.vec()->sum(), v.vec()->size());

  // operator=(const N_Vector x)
  u = v;
  ASSERT_EQ(u.vec()->sum(), u.vec()->size());

}
//----------------------------------------------------
TEST(TestSUNDIALSNVector, test_backend)
{
  // SUNDIALS 
#ifdef HAS_SUNDIALS
  _test(MPI_COMM_WORLD);
#endif
}
//----------------------------------------------------

// Test all
int SUNDIALSNVector_main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
