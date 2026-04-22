// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include "CharmmContext.h"
#include <memory>

class CharmmContext;

class CudaHolonoma {
private:
  int numSettleMolecules;
  std::vector<int4> settleiIndex;
  double tol; // float ?
  std::shared_ptr<CharmmContext> simulationContext;

public:
  CudaHolonoma();
  void setCharmmContext(std::shared_ptr<CharmmContext> context);
  void setup();

  void constrainWaterMolecules();
};
