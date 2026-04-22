// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Félix Aviat, Samarjeet Prasad
//
// ENDLICENSE

#include "CudaContainer.h"
#include "compare.h"
#include "cuda_utils.h"
#include "helper.h"

/* IDEA: test a few functions of the CudaContainer class */

TEST_CASE("unittest") {
  int size = 10;
  SECTION("setToValue") {
    int intRef = 12;
    float floatRef = 12.0;
    double4 double4Ref = make_double4(12.0, 12.0, 12.0, 12.0);
    CudaContainer<int> intContainer(size);
    CudaContainer<float> floatContainer;
    CudaContainer<double4> double4Container;
    floatContainer = CudaContainer<float>(size);
    double4Container.resize(size);
    intContainer.set(intRef);
    floatContainer.set(floatRef);
    double4Container.set(double4Ref);
    intContainer.transferFromDevice();
    floatContainer.transferFromDevice();
    double4Container.transferFromDevice();

    bool intCheck = true, floatCheck = true, double4Check = true;
    for (int i; i < size; i++) {
      if (intContainer[i] != intRef)
        intCheck = false;
      if (floatContainer[i] != floatRef)
        floatCheck = false;
      if ((double4Container[i].x != double4Ref.x) ||
          (double4Container[i].y != double4Ref.y) ||
          (double4Container[i].z != double4Ref.z) ||
          (double4Container[i].w != double4Ref.w))
        double4Check = false;
    }
    CHECK(intCheck);
    CHECK(floatCheck);
    CHECK(double4Check);
  }
}
