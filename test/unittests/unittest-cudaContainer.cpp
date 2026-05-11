// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II, Félix Aviat, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaContainer.h"
#include "catch.hpp"
#include "compare.h"
#include "cuda_utils.h"
#include <vector>

TEST_CASE("ConstructionDestruction") {
  SECTION("DefaultConstructor") {
    CudaContainer<int> c;
    CHECK(c.size() == 0);
    CHECK(c.getHostArray().empty() == true);
    CHECK(c.getDeviceArray().empty() == true);
  }

  SECTION("SizeConstructor") {
    CudaContainer<int> c(5);
    CHECK(c.size() == 5);
    CHECK(c.getHostArray().size() == 5);
    CHECK(c.getDeviceArray().size() == 5);
  }

  SECTION("HostVectorConstructor") {
    std::vector<int> v = {1, 2, 3};
    CudaContainer<int> c(v);
    CHECK(c.size() == 3);
    CHECK(c.getHostArray().size() == 3);
    CHECK(c.getDeviceArray().size() == 3);
    CHECK(CompareVectors1<int>(c.getHostArray(), v, 0.0, true));
    CHECK_NOTHROW(c.transferToHost());
    CHECK(CompareVectors1<int>(c.getHostArray(), v, 0.0, true));
  }

  SECTION("DeviceVectorConstructor") {
    DeviceVector<int> v({1, 2, 3});
    CudaContainer<int> c(v);
    CHECK(c.size() == 3);
    CHECK(c.getHostArray().size() == 3);
    CHECK(c.getDeviceArray().size() == 3);
    std::vector<int> u = {1, 2, 3};
    CHECK(CompareVectors1<int>(c.getHostArray(), u, 0.0, true));
    CHECK_NOTHROW(c.transferToHost());
    CHECK(CompareVectors1<int>(c.getHostArray(), u, 0.0, true));
  }

  SECTION("CopyConstructor") {
    CudaContainer<int> c1(std::vector<int>{1, 2, 3});
    CudaContainer<int> c2(c1);
    CHECK(c1.size() == c2.size());
    CHECK(c1.getHostArray().data() != c2.getHostArray().data());
    CHECK(c1.getDeviceArray().data() != c2.getDeviceArray().data());
    CHECK(CompareVectors1(c1.getHostArray(), c2.getHostArray(), 0.0, true));
    CHECK_NOTHROW(c1.transferToHost());
    CHECK_NOTHROW(c2.transferToHost());
    CHECK(
        CompareVectors1<int>(c1.getHostArray(), c2.getHostArray(), 0.0, true));
  }
}

TEST_CASE("Assignment") {
  SECTION("AssignFromHostVector") {
    CudaContainer<int> c;
    std::vector<int> u({1, 2, 3});
    CHECK_NOTHROW(c = u);
    CHECK(CompareVectors1<int>(c.getHostArray(), u, 0.0, true));
    CHECK_NOTHROW(c.transferToHost());
    CHECK(CompareVectors1<int>(c.getHostArray(), u, 0.0, true));
  }

  SECTION("AssignFromDeviceVector") {
    CudaContainer<int> c;
    DeviceVector<int> v({1, 2, 3});
    CHECK_NOTHROW(c = v);
    std::vector<int> u({1, 2, 3});
    CHECK(CompareVectors1<int>(c.getHostArray(), u, 0.0, true));
    CHECK_NOTHROW(c.transferToHost());
    CHECK(CompareVectors1<int>(c.getHostArray(), u, 0.0, true));
  }
}
