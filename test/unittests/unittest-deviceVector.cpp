// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "DeviceVector.h"
#include "catch.hpp"
#include "compare.h"
#include "cuda_utils.h"
#include <vector>

template <typename T> void ConstructDestroy(const std::vector<T> &u) {
  for (std::size_t i = 0; i < 10000; i++) {
    DeviceVector<int> v(u.size());
    cudaCheck(cudaMemcpy(static_cast<void *>(v.data()),
                         static_cast<const void *>(u.data()),
                         u.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  return;
}

TEST_CASE("ConstructionDestruction") {
  SECTION("DefaultConstructor") {
    DeviceVector<int> v;
    CHECK(v.empty() == true);
    CHECK(v.size() == 0);
    CHECK(v.capacity() == 0);
    CHECK(v.data() == nullptr);
  }

  SECTION("SizeConstructorZero") {
    DeviceVector<int> v(0);
    CHECK(v.empty() == true);
    CHECK(v.size() == 0);
    CHECK(v.capacity() == 0);
    CHECK_NOTHROW(v.clear());
  }

  SECTION("SizeConstructorNonzero") {
    constexpr std::size_t n = 5;
    DeviceVector<int> v(n);
    CHECK(v.empty() == false);
    CHECK(v.size() == n);
    CHECK(v.capacity() == n);
    CHECK(v.data() != nullptr);

    std::vector<int> u(n, 4);
    cudaCheck(cudaMemcpy(static_cast<void *>(v.data()),
                         static_cast<const void *>(u.data()), n * sizeof(int),
                         cudaMemcpyHostToDevice));
    u.assign(n, -1); // Ensure that we get the "right" value from GPU
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), n * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(CompareVector<int>(u, 4, 0.0, true));
  }

  SECTION("HostVectorConstructor") {
    std::vector<int> u = {1, 2, 3, 4};
    DeviceVector<int> v(u);
    u.assign(4, -1);
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 1);
    CHECK(u[1] == 2);
    CHECK(u[2] == 3);
    CHECK(u[3] == 4);
  }

  SECTION("CopyConstructor") {
    DeviceVector<int> v1({1, 2, 3, 4, 5});
    DeviceVector<int> v2(v1);
    CHECK(v1.data() != v2.data());
    std::vector<int> u1(5, -1); // Ensure that we get the "right" value from GPU
    cudaCheck(cudaMemcpy(static_cast<void *>(v1.data()),
                         static_cast<const void *>(u1.data()), sizeof(int),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(static_cast<void *>(u1.data()),
                         static_cast<const void *>(v1.data()), 5 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u1[0] == -1);
    CHECK(u1[1] == 2);
    CHECK(u1[2] == 3);
    CHECK(u1[3] == 4);
    CHECK(u1[4] == 5);
    std::vector<int> u2(5, -1);
    cudaCheck(cudaMemcpy(static_cast<void *>(u2.data()),
                         static_cast<const void *>(v2.data()), 5 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u2[0] == 1);
    CHECK(u2[1] == 2);
    CHECK(u2[2] == 3);
    CHECK(u2[3] == 4);
    CHECK(u2[4] == 5);
  }

  SECTION("RepeatedConstructDestroy") {
    std::vector<int> u(1024);
    for (std::size_t i = 0; i < 1024; i++)
      u[i] = i + 1;
    CHECK_NOTHROW(ConstructDestroy<int>(u));
  }
}

TEST_CASE("CapacityAndResizeBehavior") {
  SECTION("ResizeFromEmpty") {
    constexpr std::size_t n = 5;
    DeviceVector<int> v;
    v.resize(n);
    CHECK(v.size() == n);
    CHECK(v.capacity() == n);
    CHECK(v.data() != nullptr);
  }

  SECTION("ResizeGrowPreserve") {
    DeviceVector<int> v({1, 2, 3});
    v.resize(8);
    std::vector<int> u(8, -1); // Ensure that we get the "right" value from GPU
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 8 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 1);
    CHECK(u[1] == 2);
    CHECK(u[2] == 3);
  }

  SECTION("ResizeShrinkPreserve") {
    DeviceVector<int> v({1, 2, 3, 4, 5, 6});
    v.resize(3);
    std::vector<int> u(3, -1); // Ensure that we get the "right" value from GPU
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 3 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 1);
    CHECK(u[1] == 2);
    CHECK(u[2] == 3);
  }

  SECTION("ShrinkToFitNonempty") {
    DeviceVector<int> v({1, 2, 3, 4, 5, 6});
    v.resize(3);
    v.shrink_to_fit();
    std::vector<int> u(3, -1); // Ensure that we get the "right" value from GPU
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 3 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 1);
    CHECK(u[1] == 2);
    CHECK(u[2] == 3);
  }

  SECTION("ShrinkToFitEmpty") {
    DeviceVector<int> v;
    CHECK_NOTHROW(v.shrink_to_fit());
    CHECK(v.size() == 0);
    CHECK(v.capacity() == 0);
  }

  SECTION("Clear") {
    DeviceVector<int> v({1, 2, 3, 4, 5, 6});
    v.clear();
    CHECK(v.size() == 0);
    CHECK(v.capacity() == 0);
    CHECK_NOTHROW(v.clear());
    CHECK(v.size() == 0);
    CHECK(v.capacity() == 0);
  }

  SECTION("PushBackFromEmpty") {
    DeviceVector<int> v;
    v.push_back(4);
    CHECK(v.size() == 1);
    CHECK(v.capacity() >= 1);
    std::vector<int> u(1);
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 4);
  }

  SECTION("PushBackSecondElement") {
    DeviceVector<int> v(std::vector<int>(1, 1));
    v.push_back(4);
    CHECK(v.size() == 2);
    CHECK(v.capacity() >= 2);
    std::vector<int> u(2, -1); // Ensure that we get the "right" value from GPU
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 2 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 1);
    CHECK(u[1] == 4);
  }

  SECTION("PushBackMany") {
    std::vector<int> expected(100);
    for (int i = 0; i < 100; i++)
      expected[i] = i + 1;

    DeviceVector<int> v;
    for (int i = 0; i < 100; i++)
      v.push_back(i + 1);

    CHECK(v.size() == 100);
    CHECK(v.capacity() >= 100);

    std::vector<int> u(100, -1);
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 100 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(CompareVectors1<int>(u, expected, 0.0, true));
  }

  SECTION("PushBackAfterShrink") {
    DeviceVector<int> v(8);
    std::vector<int> u = {1, 2, 3, 4};
    cudaCheck(cudaMemcpy(static_cast<void *>(v.data()),
                         static_cast<const void *>(u.data()), 4 * sizeof(int),
                         cudaMemcpyHostToDevice));
    v.resize(4);
    v.shrink_to_fit();
    v.push_back(64);
    u.resize(5);
    u.assign(5, -1);
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 5 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 1);
    CHECK(u[1] == 2);
    CHECK(u[2] == 3);
    CHECK(u[3] == 4);
    CHECK(u[4] == 64);
  }
}

TEST_CASE("CopyAssignmentSwap") {
  SECTION("VectorAssign") {
    std::vector<int> u = {1, 2, 3, 4};
    DeviceVector<int> v;
    v = u;
    u.assign(4, -1);
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 1);
    CHECK(u[1] == 2);
    CHECK(u[2] == 3);
    CHECK(u[3] == 4);
  }

  SECTION("CopyAssignmentDeepCopy") {
    DeviceVector<int> v1({1, 2, 3, 4});
    DeviceVector<int> v2(100);
    DeviceVector<int> v3(2);
    v2 = v1;
    v3 = v1;
    CHECK(v1.size() == v2.size());
    CHECK(v1.size() == v3.size());
    CHECK(v1.capacity() == v2.capacity());
    CHECK(v1.capacity() == v3.capacity());
    // Ensure that we get the "right" value from GPU
    std::vector<int> u1(4, -1), u2(4, -2), u3(4, -3);
    cudaCheck(cudaMemcpy(static_cast<void *>(u1.data()),
                         static_cast<const void *>(v1.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(static_cast<void *>(u2.data()),
                         static_cast<const void *>(v2.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(static_cast<void *>(u3.data()),
                         static_cast<const void *>(v3.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(CompareVectors1<int>(u1, u2, 0.0, true));
    CHECK(CompareVectors1<int>(u1, u3, 0.0, true));
  }

  SECTION("RvalueConstructorCopySemantics") {
    DeviceVector<int> v1({1, 2, 3, 4});
    DeviceVector<int> v2(std::move(v1));
    CHECK(v1.size() == v2.size());
    CHECK(v1.capacity() == v2.capacity());
    // Ensure that we get the "right" value from GPU
    std::vector<int> u1(4, -1), u2(4, -2);
    cudaCheck(cudaMemcpy(static_cast<void *>(u1.data()),
                         static_cast<const void *>(v1.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(static_cast<void *>(u2.data()),
                         static_cast<const void *>(v2.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(CompareVectors1<int>(u1, u2, 0.0, true));
  }

  SECTION("SelfAssignment") {
    DeviceVector<int> v({1, 2, 3, 4});
    CHECK_NOTHROW(v = v);
    std::vector<int> u(4, -1); // Ensure that we get the "right" value from GPU
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 1);
    CHECK(u[1] == 2);
    CHECK(u[2] == 3);
    CHECK(u[3] == 4);
  }

  SECTION("SelfMoveAssignment") {
    DeviceVector<int> v({1, 2, 3, 4});
    CHECK_NOTHROW(v = std::move(v));
    std::vector<int> u(4, -1); // Ensure that we get the "right" value from GPU
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 1);
    CHECK(u[1] == 2);
    CHECK(u[2] == 3);
    CHECK(u[3] == 4);
  }

  SECTION("SwapNonemptyNonempty") {
    DeviceVector<int> v1({1, 2, 3, 4});
    DeviceVector<int> v2({5, 4, 3, 2, 1});
    CHECK_NOTHROW(v1.swap(v2));
    CHECK(v1.size() == 5);
    CHECK(v1.capacity() == 5);
    CHECK(v2.size() == 4);
    CHECK(v2.capacity() == 4);
    // Ensure that we get the "right" value from GPU
    std::vector<int> u1(5, -1), u2(4, -1);
    cudaCheck(cudaMemcpy(static_cast<void *>(u1.data()),
                         static_cast<const void *>(v1.data()), 5 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(static_cast<void *>(u2.data()),
                         static_cast<const void *>(v2.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u1[0] == 5);
    CHECK(u1[1] == 4);
    CHECK(u1[2] == 3);
    CHECK(u1[3] == 2);
    CHECK(u1[4] == 1);
    CHECK(u2[0] == 1);
    CHECK(u2[1] == 2);
    CHECK(u2[2] == 3);
    CHECK(u2[3] == 4);
  }

  SECTION("SwapEmptyNonempty") {
    DeviceVector<int> v1;
    DeviceVector<int> v2({1, 2, 3, 4});
    CHECK_NOTHROW(v1.swap(v2));
    CHECK(v1.size() == 4);
    CHECK(v1.capacity() == 4);
    CHECK(v2.size() == 0);
    CHECK(v2.capacity() == 0);
    std::vector<int> u1(4, -1); // Ensure that we get the "right" value from GPU
    cudaCheck(cudaMemcpy(static_cast<void *>(u1.data()),
                         static_cast<const void *>(v1.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u1[0] == 1);
    CHECK(u1[1] == 2);
    CHECK(u1[2] == 3);
    CHECK(u1[3] == 4);
  }

  SECTION("SwapSelf") {
    DeviceVector<int> v({1, 2, 3, 4});
    CHECK_NOTHROW(v.swap(v));
    CHECK(v.size() == 4);
    CHECK(v.capacity() == 4);
    std::vector<int> u(4, -1); // Ensure that we get the "right" value from GPU
    cudaCheck(cudaMemcpy(static_cast<void *>(u.data()),
                         static_cast<const void *>(v.data()), 4 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CHECK(u[0] == 1);
    CHECK(u[1] == 2);
    CHECK(u[2] == 3);
    CHECK(u[3] == 4);
  }
}
