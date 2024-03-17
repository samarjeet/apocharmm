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
#include "helper.h"

/* IDEA: test a few functions of the CudaContainer class */

TEST_CASE("unittest") {
  int size = 10;
  SECTION("setToValue") {
    int intRef = 12;
    float floatRef = 12.0;
    double4 double4Ref = {12.0, 12.0, 12.0, 12.0};
    CudaContainer<int> intContainer;
    CudaContainer<float> floatContainer;
    CudaContainer<double4> double4Container;
    intContainer.allocate(size);
    floatContainer.allocate(size);
    double4Container.allocate(size);
    intContainer.setToValue(intRef);
    floatContainer.setToValue(floatRef);
    double4Container.setToValue(double4Ref);
    intContainer.transferFromDevice();
    floatContainer.transferFromDevice();
    double4Container.transferFromDevice();

    bool intCheck = true, floatCheck = true, double4Check = true;
    for (int i; i < size; i++) {
      if (intContainer.getHostArray()[i] != intRef)
        intCheck = false;
      if (floatContainer.getHostArray()[i] != floatRef)
        floatCheck = false;
      if ((double4Container.getHostArray()[i].x != double4Ref.x) or //
          (double4Container.getHostArray()[i].y != double4Ref.y) or //
          (double4Container.getHostArray()[i].z != double4Ref.z) or //
          (double4Container.getHostArray()[i].w != double4Ref.w))
        double4Check = false;
    }
    CHECK(intCheck);
    CHECK(floatCheck);
    CHECK(double4Check);
  }
}