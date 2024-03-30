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
// #include "CharmmContext.h"
// #include "CudaHolonoma.h"
#include <memory>
#include <string>

class CharmmContext;

class CudaMinimizer {
private:
  int nsteps;
  std::shared_ptr<CharmmContext> context;
  std::string method;
  bool verboseFlag;

public:
  CudaMinimizer();

  // This should not be a raw pointer
  void setCharmmContext(std::shared_ptr<CharmmContext> csc);
  void minimize(int numSteps);
  void minimize();

  void setVerboseFlag(bool _flag = true);
};
